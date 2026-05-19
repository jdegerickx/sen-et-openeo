"""Validate pyTSEB-PT latent heat flux (LE, W/m²) against ICOS tower
measurements.

Design
------
The TSEB-PT pipeline runs on full Sentinel-2 tiles because the sharpener is
trained on tile-wide S2/S3 data — it cannot be re-run on a small bbox around
each tower.  This script therefore:

  1. **Loads ICOS sites** from the ZIP archives (coordinates + half-hourly
     LE_F_MDS flux, quality-filtered by LE_F_MDS_QC).

  2. **Determines the S2 tile(s) that contain each tower** using the bundled
     S2 grid GeoJSON and looks for already-processed TSEB-PT VRT files under
     ``output_root / <tile> / <et_subdir>/``.  A clear warning is printed for
     any tower whose tile has not been processed yet.

  3. **Temporal matching** — for each VRT, the overpass UTC time is parsed
     from the filename (``TSEB-PT_YYYYMMDDTHHMMSS_<tile>.vrt``) and the ICOS
     half-hour window that contains it is selected.

  4. **Spatial extraction** — the ``LE1`` band (instantaneous LE, W/m²) from
     the per-field TIF sibling is sampled as the mean of all valid pixels
     within ``extraction_radius_m`` metres of the tower.

  5. **Metrics** (per site and overall):
       bias, MAE, RMSE, Pearson r, r², Nash–Sutcliffe E

  6. **Figures** saved in ``output_dir``:
       scatter_per_site.png, scatter_all_sites.png,
       timeseries_per_site.png, metrics_bar.png
     Plus ``paired_le_obs_mod.csv`` and ``metrics_table.csv``.

Usage
-----
Run as a standalone script (edit the settings block at the bottom) or import
:func:`main` from another script.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from loguru import logger
from scipy import stats
from shapely.geometry import Point, shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LE_OBS_COL = "LE_F_MDS"  # observed latent heat flux column
LE_QC_COL = "LE_F_MDS_QC"  # quality-control column (0 = measured)
H_OBS_COL = "H_F_MDS"  # sensible heat flux column
RN_OBS_COL = "NETRAD"  # net radiation column
G_OBS_COL = "G_F_MDS"  # ground heat flux column
ICOS_NODATA = -9999.0  # ICOS / FLUXNET missing-value sentinel

_SITEINFO_PAT = re.compile(r"ICOSETC_.+?_SITEINFO.*\.csv", re.I)
_FLUXNET_HH_PAT = re.compile(r"ICOSETC_.+?_FLUXNET_HH.*\.csv", re.I)
_VRT_PAT = re.compile(r"TSEB-PT_(\d{8}T\d{6})_(\w+)\.vrt$", re.I)

# Path to the bundled S2 grid GeoJSON (ships with sen_et_openeo)
_S2GRID_GEOJSON = (
    Path(__file__).parent.parent
    / "src"
    / "sen_et_openeo"
    / "layers"
    / "s2grid_bounds.geojson"
)

# ---------------------------------------------------------------------------
# Step 1 – Load ICOS sites from ZIP archives
# ---------------------------------------------------------------------------


def _read_csv_from_zip(zf: zipfile.ZipFile, pattern: re.Pattern) -> pd.DataFrame:
    for name in zf.namelist():
        if pattern.search(name):
            with zf.open(name) as fh:
                return pd.read_csv(
                    io.TextIOWrapper(fh, encoding="utf-8-sig"),
                    on_bad_lines="skip",
                )
    raise FileNotFoundError(f"No entry matching {pattern.pattern!r} in {zf.filename}")


def _parse_siteinfo(df: pd.DataFrame) -> dict:
    info: dict = {}
    for _, row in df.iterrows():
        var = str(row.get("VARIABLE", "")).strip()
        val = row.get("DATAVALUE", None)
        if var in (
            "LOCATION_LAT",
            "LOCATION_LONG",
            "LOCATION_ELEV",
            "IGBP",
            "SITE_NAME",
        ):
            try:
                info[var] = float(val)
            except (ValueError, TypeError):
                info[var] = str(val).strip()
    return info


def load_icos_sites(icos_archive_dir: Path, max_qc: int = 0) -> Dict[str, dict]:
    """Load all ICOS sites from ZIP archives in *icos_archive_dir*.

    Parameters
    ----------
    icos_archive_dir:
        Folder containing per-site ``ICOSETC_*_ARCHIVE*.zip`` files.
    max_qc:
        Maximum allowed ``LE_F_MDS_QC`` (0 = directly measured only).

    Returns
    -------
    dict  site_id -> {lat, lon, elev, igbp, name, df}
        where *df* has columns [TIMESTAMP_START, TIMESTAMP_END, LE_F_MDS,
        LE_F_MDS_QC] filtered to QC <= max_qc, timestamps in UTC.
    """
    archive_dir = Path(icos_archive_dir)
    zip_files = sorted(archive_dir.rglob("ICOSETC_*_ARCHIVE*.zip"))
    if not zip_files:
        zip_files = sorted(archive_dir.rglob("*.zip"))

    sites: Dict[str, dict] = {}
    for zpath in zip_files:
        logger.info(f"Loading ICOS site from {zpath.name}")
        with zipfile.ZipFile(zpath, "r") as zf:
            meta = _parse_siteinfo(_read_csv_from_zip(zf, _SITEINFO_PAT))
            hh_df = _read_csv_from_zip(zf, _FLUXNET_HH_PAT)

        m = re.search(r"ICOSETC_(.+?)_ARCHIVE", zpath.name, re.I)
        site_id = m.group(1) if m else zpath.stem

        lat = float(meta.get("LOCATION_LAT", np.nan))
        lon = float(meta.get("LOCATION_LONG", np.nan))
        elev = meta.get("LOCATION_ELEV", np.nan)
        try:
            elev = float(elev)
            if elev == ICOS_NODATA:
                elev = np.nan
        except (ValueError, TypeError):
            elev = np.nan

        # Parse timestamps
        hh_df["TIMESTAMP_START"] = pd.to_datetime(
            hh_df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M", utc=True
        )
        hh_df["TIMESTAMP_END"] = pd.to_datetime(
            hh_df["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M", utc=True
        )

        # Replace nodata sentinels
        optional_cols = [H_OBS_COL, RN_OBS_COL, G_OBS_COL]
        for col in (LE_OBS_COL, LE_QC_COL, *optional_cols):
            if col in hh_df.columns:
                hh_df[col] = pd.to_numeric(hh_df[col], errors="coerce")
                hh_df.loc[hh_df[col] == ICOS_NODATA, col] = np.nan

        required = ["TIMESTAMP_START", "TIMESTAMP_END", LE_OBS_COL, LE_QC_COL]
        missing = [c for c in required if c not in hh_df.columns]
        if missing:
            logger.warning(f"{site_id}: missing columns {missing}, skipping")
            continue

        keep_cols = required + [c for c in optional_cols if c in hh_df.columns]
        hh_df = (
            hh_df[keep_cols]
            .dropna(subset=[LE_OBS_COL, LE_QC_COL])
            .query(f"{LE_QC_COL} <= {max_qc}")
            .reset_index(drop=True)
        )

        logger.info(
            f"  {site_id}: lat={lat:.4f}, lon={lon:.4f}, "
            f'IGBP={meta.get("IGBP","?")}, N_obs={len(hh_df)} (QC<={max_qc})'
        )

        sites[site_id] = {
            "lat": lat,
            "lon": lon,
            "elev": elev,
            "igbp": meta.get("IGBP", "unknown"),
            "name": meta.get("SITE_NAME", ""),
            "df": hh_df,
        }
    return sites


# ---------------------------------------------------------------------------
# Step 2 – Map towers to S2 tiles using the grid GeoJSON
# ---------------------------------------------------------------------------


def _load_s2_grid() -> list:
    """Return list of (tile_name, shapely_geometry) from the bundled GeoJSON."""
    if not _S2GRID_GEOJSON.exists():
        raise FileNotFoundError(f"S2 grid GeoJSON not found: {_S2GRID_GEOJSON}")
    with open(_S2GRID_GEOJSON) as fh:
        grid = json.load(fh)
    return [(f["properties"]["tile"], shape(f["geometry"])) for f in grid["features"]]


def find_tiles_for_tower(lat: float, lon: float, s2_grid: list) -> List[str]:
    """Return all S2 tile names whose footprint contains (lon, lat)."""
    pt = Point(lon, lat)
    return [tile for tile, geom in s2_grid if geom.contains(pt)]


# ---------------------------------------------------------------------------
# Step 3 – Temporal matching
# ---------------------------------------------------------------------------


def _find_icos_record(
    overpass_utc: pd.Timestamp, site_df: pd.DataFrame
) -> Optional[pd.Series]:
    """Return the ICOS half-hour whose window contains *overpass_utc*."""
    mask = (site_df["TIMESTAMP_START"] <= overpass_utc) & (
        overpass_utc < site_df["TIMESTAMP_END"]
    )
    rows = site_df[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return None if pd.isna(row[LE_OBS_COL]) else row


def _bowen_ratio_le(row: pd.Series, eps: float = 1e-6) -> float:
    """Return LE closed by Bowen-ratio energy-balance closure.

    Uses beta = H / LE and enforces closure with:
      LE_br = (Rn - G) / (1 + beta)

    Returns NaN when required fluxes are missing or numerically unstable.
    """
    le = row.get(LE_OBS_COL, np.nan)
    h = row.get(H_OBS_COL, np.nan)
    rn = row.get(RN_OBS_COL, np.nan)
    g = row.get(G_OBS_COL, np.nan)

    if not np.isfinite(le) or not np.isfinite(h):
        return np.nan
    if not np.isfinite(rn) or not np.isfinite(g):
        return np.nan
    if abs(le) < eps:
        return np.nan

    beta = h / le
    denom = 1.0 + beta
    if not np.isfinite(denom) or abs(denom) < eps:
        return np.nan

    return float((rn - g) / denom)


def _le_to_et_mm_h(le_w_m2: pd.Series, lambda_v_j_kg: float = 2.45e6) -> pd.Series:
    """Convert latent heat flux (W/m2) to instantaneous ET rate (mm/h)."""
    return (le_w_m2 / lambda_v_j_kg) * 3600.0


def _calculate_daily_et_from_le(
    le_w_m2_series: pd.Series, timestamps: pd.Series, lambda_v_j_kg: float = 2.45e6
) -> dict:
    """Aggregate instantaneous LE measurements to daily ET (mm/day).

    Uses daily mean LE scaled to 24 hours per calendar day (UTC).

    Parameters
    ----------
    le_w_m2_series:
        Instantaneous LE values (W/m²)
    timestamps:
        UTC timestamps corresponding to each LE value
    lambda_v_j_kg:
        Latent heat of vaporization (J/kg), default 2.45e6

    Returns
    -------
    dict  {date_str -> daily_et_mm}
        Mapping of YYYY-MM-DD to daily ET in mm/day
    """
    df_tmp = pd.DataFrame({"timestamp": timestamps, "le": le_w_m2_series}).sort_values(
        "timestamp"
    )

    # Add date column
    df_tmp["date"] = df_tmp["timestamp"].dt.date

    daily_et = {}
    for date, grp in df_tmp.groupby("date"):
        # Daily-mean scaling to 24 h.
        le_mean = grp["le"].mean()
        if np.isfinite(le_mean):
            # Convert W/m² to mm/day: LE / lambda × 86400.
            # No extra 1/1000 factor is needed.
            et_mm_day = (le_mean / lambda_v_j_kg) * 86400.0
            daily_et[str(date)] = float(et_mm_day)
        else:
            daily_et[str(date)] = np.nan

    return daily_et


# ---------------------------------------------------------------------------
# Step 4 – Spatial extraction from TSEB VRT
# ---------------------------------------------------------------------------


def extract_le_at_tower(
    vrt_file: Path,
    lat: float,
    lon: float,
    radius_m: float = 150.0,
    le_band_name: str = "LE1",
) -> Optional[float]:
    """Return mean LE (W/m2) within *radius_m* metres of the tower.

    Reads ``<vrt_stem>.data/LE1.tif`` (the per-field TIF written alongside
    each VRT by _process_tseb_tiled).  Projects the WGS-84 tower coordinates
    into the raster CRS before sampling.

    Returns None if the tower is outside the raster or no valid pixels are
    found inside the radius.
    """
    vrt_file = Path(vrt_file)
    le_tif = vrt_file.parent / (vrt_file.stem + ".data") / f"{le_band_name}.tif"

    if not le_tif.exists():
        logger.warning(f"LE TIF not found: {le_tif}")
        return None

    with rasterio.open(le_tif) as src:
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata
        width, height = src.width, src.height

        # Project tower -> raster CRS
        tr = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        x_m, y_m = tr.transform(lon, lat)

        col_f, row_f = ~transform * (x_m, y_m)
        if not (0 <= col_f < width and 0 <= row_f < height):
            logger.debug(f"Tower ({lat:.4f},{lon:.4f}) outside raster {le_tif.name}")
            return None

        px_size = abs(transform.a)
        radius_px = radius_m / px_size
        pad = int(np.ceil(radius_px)) + 1

        col_c = int(col_f)
        row_c = int(row_f)
        c0 = max(0, col_c - pad)
        c1 = min(width, col_c + pad + 1)
        r0 = max(0, row_c - pad)
        r1 = min(height, row_c + pad + 1)

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(1, window=window).astype(np.float64)

    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    circle = np.sqrt((cc - col_f) ** 2 + (rr - row_f) ** 2) <= radius_px

    valid = circle & np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata

    if not np.any(valid):
        logger.debug(f"No valid pixels within {radius_m} m of ({lat:.4f},{lon:.4f})")
        return None

    return float(np.mean(data[valid]))


def extract_et_day_at_tower(
    vrt_file: Path,
    lat: float,
    lon: float,
    radius_m: float = 150.0,
    et_day_band_name: str = "ET_day",
) -> Optional[float]:
    """Return mean daily ET (mm/day) within *radius_m* metres of the tower.

    Reads ``<vrt_stem>.data/ET_day.tif`` (pyTSEB default daily ET output).
    If not found, tries a small fallback list of historical/alternate names.
    Returns None if no daily ET band is available.

    Parameters
    ----------
    vrt_file:
        Path to TSEB-PT VRT file
    lat, lon:
        Tower coordinates (WGS-84)
    radius_m:
        Extraction radius in metres
    et_day_band_name:
        Preferred daily ET band name. Defaults to ``ET_day``.

    Returns
    -------
    float or None
        Mean daily ET (mm/day) within radius, or None if not found
    """
    vrt_file = Path(vrt_file)
    data_dir = vrt_file.parent / (vrt_file.stem + ".data")
    band_candidates = [et_day_band_name, "ET_day", "ET1_24", "ET_24h"]
    # Keep order while removing duplicates.
    band_candidates = list(dict.fromkeys(band_candidates))

    et_day_tif = None
    for band_name in band_candidates:
        candidate = data_dir / f"{band_name}.tif"
        if candidate.exists():
            et_day_tif = candidate
            break

    if et_day_tif is None:
        logger.debug(
            f"Daily ET TIF not found in {data_dir}; " f"tried {band_candidates}"
        )
        return None

    try:
        with rasterio.open(et_day_tif) as src:
            raster_crs = src.crs
            transform = src.transform
            nodata = src.nodata
            width, height = src.width, src.height

            # Project tower -> raster CRS
            tr = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            x_m, y_m = tr.transform(lon, lat)

            col_f, row_f = ~transform * (x_m, y_m)
            if not (0 <= col_f < width and 0 <= row_f < height):
                logger.debug(
                    f"Tower ({lat:.4f},{lon:.4f}) outside raster {et_day_tif.name}"
                )
                return None

            px_size = abs(transform.a)
            radius_px = radius_m / px_size
            pad = int(np.ceil(radius_px)) + 1

            col_c = int(col_f)
            row_c = int(row_f)
            c0 = max(0, col_c - pad)
            c1 = min(width, col_c + pad + 1)
            r0 = max(0, row_c - pad)
            r1 = min(height, row_c + pad + 1)

            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=window).astype(np.float64)

        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        circle = np.sqrt((cc - col_f) ** 2 + (rr - row_f) ** 2) <= radius_px

        valid = circle & np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata

        if not np.any(valid):
            logger.debug(
                f"No valid pixels within {radius_m} m of ({lat:.4f},{lon:.4f})"
            )
            return None

        return float(np.mean(data[valid]))
    except Exception as e:
        logger.debug(f"Error reading daily ET for tower: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 5 – Collect all matched pairs
# ---------------------------------------------------------------------------


def collect_paired_data(
    sites: Dict[str, dict],
    output_root: Union[Path, List[Path]],
    et_subdirs: List[str],
    s2_grid: list,
    extraction_radius_m: float = 150.0,
    le_band_name: str = "LE1",
    min_daily_obs: int = 24,
    drop_negative_daily_obs: bool = True,
) -> pd.DataFrame:
    """Build a DataFrame of (observed, modelled) LE pairs.

    For each ICOS site the function:
      - finds which S2 tile(s) contain the tower,
      - scans output_root / tile / et_subdir for TSEB-PT VRTs,
      - warns clearly if no VRTs are found (tile not yet processed),
      - matches each VRT overpass time to the ICOS half-hour record,
      - extracts the mean LE within the extraction radius.

    Parameters
    ----------
    sites:
        Output of load_icos_sites.
    output_root:
        Root of the pyTSEB output tree, or list of roots (e.g. multiple years).
    et_subdirs:
        Sub-directory names under each tile to scan
        (e.g. ['007_et', '008_lstm-et']).
    s2_grid:
        Output of _load_s2_grid.
    extraction_radius_m:
        Circular extraction radius in metres.
    le_band_name:
        Band name for instantaneous LE in the .data/ folder.
    min_daily_obs:
        Minimum number of valid half-hourly LE records required to compute
        an observed ET_day value for a site/date. Default 24.
    drop_negative_daily_obs:
        If True, negative observed ET_day values are replaced by NaN and
        logged as potential condensation/closure outliers.

    Returns
    -------
    pd.DataFrame with columns:
        site_id, igbp, et_product, tile, datetime_utc,
        LE_obs (W/m2), LE_mod (W/m2), vrt_file
    """
    records = []

    # Accept one root path or multiple roots (e.g., 2024 + 2025) in one run.
    if isinstance(output_root, (str, Path)):
        output_roots = [Path(output_root)]
    else:
        output_roots = [Path(p) for p in output_root]

    for site_id, site in sites.items():
        tiles = find_tiles_for_tower(site["lat"], site["lon"], s2_grid)
        if not tiles:
            logger.warning(
                f"{site_id}: no S2 tile found for "
                f'({site["lat"]:.4f}, {site["lon"]:.4f}), skipping'
            )
            continue
        logger.info(f"{site_id} falls in tile(s): {tiles}")

        for output_root_i in output_roots:
            for tile in tiles:
                for et_sub in et_subdirs:
                    et_dir = output_root_i / tile / et_sub
                    if not et_dir.is_dir():
                        logger.warning(
                            f"  {site_id} / {tile} / {et_sub}: directory not found under "
                            f"{output_root_i} — run run_sen-et.py for tile {tile} first"
                        )
                        continue

                    vrts = sorted(et_dir.glob("TSEB-PT_*.vrt"))
                    if not vrts:
                        logger.warning(
                            f"  {site_id} / {tile} / {et_sub}: no TSEB-PT VRTs found under "
                            f"{output_root_i} — run run_sen-et.py for tile {tile} first"
                        )
                        continue

                    logger.info(
                        f"  {site_id} / {tile} / {et_sub} / {output_root_i.name}: "
                        f"scanning {len(vrts)} VRTs"
                    )

                    for vrt in vrts:
                        m = _VRT_PAT.search(vrt.name)
                        if not m:
                            continue
                        overpass_utc = pd.Timestamp(m.group(1), tz="UTC")

                        row = _find_icos_record(overpass_utc, site["df"])
                        if row is None:
                            continue

                        le_mod = extract_le_at_tower(
                            vrt,
                            site["lat"],
                            site["lon"],
                            radius_m=extraction_radius_m,
                            le_band_name=le_band_name,
                        )
                        if le_mod is None:
                            continue

                        # Try to extract daily ET from TSEB output
                        et_day_mod = extract_et_day_at_tower(
                            vrt,
                            site["lat"],
                            site["lon"],
                            radius_m=extraction_radius_m,
                            et_day_band_name="ET_day",
                        )

                        records.append(
                            {
                                "site_id": site_id,
                                "igbp": site["igbp"],
                                "et_product": et_sub,
                                "tile": tile,
                                "year_root": output_root_i.name,
                                "datetime_utc": overpass_utc,
                                "LE_obs": float(row[LE_OBS_COL]),
                                "H_obs": (
                                    float(row[H_OBS_COL])
                                    if H_OBS_COL in row and pd.notna(row[H_OBS_COL])
                                    else np.nan
                                ),
                                "Rn_obs": (
                                    float(row[RN_OBS_COL])
                                    if RN_OBS_COL in row and pd.notna(row[RN_OBS_COL])
                                    else np.nan
                                ),
                                "G_obs": (
                                    float(row[G_OBS_COL])
                                    if G_OBS_COL in row and pd.notna(row[G_OBS_COL])
                                    else np.nan
                                ),
                                "LE_obs_BR": _bowen_ratio_le(row),
                                "LE_mod": le_mod,
                                "ET_day_mod": et_day_mod,
                                "vrt_file": str(vrt),
                            }
                        )
                        logger.debug(
                            f"    {site_id} {overpass_utc:%Y-%m-%dT%H:%M} "
                            f"({output_root_i.name}): "
                            f"obs={row[LE_OBS_COL]:.1f}  mod={le_mod:.1f} W/m2"
                            f'{f"  daily_et_mod={et_day_mod:.2f} mm/day" if pd.notna(et_day_mod) else ""}'
                        )

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No matched pairs found.")
    else:
        logger.info(
            f"Collected {len(df)} pairs across "
            f'{df["site_id"].nunique()} site(s), '
            f'{df["et_product"].nunique()} product(s)'
        )

    # Aggregate daily ET from all observations per site/date/product
    daily_records = []
    for (site_id, et_product), grp_keys in df.groupby(["site_id", "et_product"]):
        site = sites[site_id]
        for date, grp_day in grp_keys.groupby(grp_keys["datetime_utc"].dt.date):
            # Collect all LE observations for this calendar day
            obs_ics_records = site["df"][site["df"]["TIMESTAMP_START"].dt.date == date]

            if obs_ics_records.empty:
                continue

            # Calculate daily ET from observed LE (standard + Bowen-ratio closure)
            le_obs_arr = obs_ics_records[LE_OBS_COL].dropna()
            le_obs_br_arr = obs_ics_records.apply(
                lambda row: _bowen_ratio_le(row), axis=1
            ).dropna()

            et_obs_daily_map = (
                _calculate_daily_et_from_le(
                    le_obs_arr, obs_ics_records.loc[le_obs_arr.index, "TIMESTAMP_START"]
                )
                if not le_obs_arr.empty
                else {}
            )
            et_obs_br_daily_map = (
                _calculate_daily_et_from_le(
                    le_obs_br_arr,
                    obs_ics_records.loc[le_obs_br_arr.index, "TIMESTAMP_START"],
                )
                if not le_obs_br_arr.empty
                else {}
            )

            et_obs_day = et_obs_daily_map.get(str(date), np.nan)
            et_obs_br_day = et_obs_br_daily_map.get(str(date), np.nan)

            if len(le_obs_arr) < min_daily_obs:
                et_obs_day = np.nan
            if len(le_obs_br_arr) < min_daily_obs:
                et_obs_br_day = np.nan

            if drop_negative_daily_obs:
                if np.isfinite(et_obs_day) and et_obs_day < 0:
                    logger.warning(
                        f"{site_id} {date} {et_product}: negative ET_obs_mm_day "
                        f"({et_obs_day:.3f}) dropped to NaN "
                        f"(n={len(le_obs_arr)})"
                    )
                    et_obs_day = np.nan
                if np.isfinite(et_obs_br_day) and et_obs_br_day < 0:
                    logger.warning(
                        f"{site_id} {date} {et_product}: negative ET_obs_BR_mm_day "
                        f"({et_obs_br_day:.3f}) dropped to NaN "
                        f"(n={len(le_obs_br_arr)})"
                    )
                    et_obs_br_day = np.nan

            # Get modelled daily ET for this day (should be consistent per day)
            et_mod_day = grp_day["ET_day_mod"].dropna()
            et_mod = et_mod_day.iloc[0] if not et_mod_day.empty else np.nan

            if not le_obs_arr.empty or not le_obs_br_arr.empty:
                daily_records.append(
                    {
                        "site_id": site_id,
                        "et_product": et_product,
                        "date": str(date),
                        "ET_obs_mm_day": et_obs_day,
                        "ET_obs_BR_mm_day": et_obs_br_day,
                        "ET_mod_mm_day": et_mod,
                        "n_le_obs": len(le_obs_arr),
                        "n_le_obs_br": len(le_obs_br_arr),
                        "igbp": site["igbp"],
                    }
                )

    if daily_records:
        df_daily = pd.DataFrame(daily_records)
        logger.info(f"Aggregated {len(df_daily)} daily records")
    else:
        df_daily = pd.DataFrame()
        logger.warning("No daily records aggregated")

    # Add daily data as attribute for later use
    df.attrs["daily_df"] = df_daily

    return df


# ---------------------------------------------------------------------------
# Step 6 – Metrics
# ---------------------------------------------------------------------------


def compute_metrics(obs: np.ndarray, mod: np.ndarray) -> dict:
    n = len(obs)
    if n < 2:
        return dict(
            N=n, bias=np.nan, MAE=np.nan, RMSE=np.nan, r=np.nan, r2=np.nan, NSE=np.nan
        )
    diff = mod - obs
    r, _ = stats.pearsonr(obs, mod)
    obs_mean = obs.mean()
    nse = 1.0 - diff.var() * n / np.sum((obs - obs_mean) ** 2)
    return dict(
        N=n,
        bias=float(diff.mean()),
        MAE=float(np.abs(diff).mean()),
        RMSE=float(np.sqrt((diff**2).mean())),
        r=float(r),
        r2=float(r**2),
        NSE=float(nse),
    )


def compute_all_metrics(
    paired_df: pd.DataFrame, obs_col: str = "LE_obs", mod_col: str = "LE_mod"
) -> pd.DataFrame:
    rows = []
    for (site_id, et_product), grp in paired_df.groupby(["site_id", "et_product"]):
        valid = grp[[obs_col, mod_col]].dropna()
        m = compute_metrics(valid[obs_col].values, valid[mod_col].values)
        m.update(site_id=site_id, et_product=et_product, igbp=grp["igbp"].iloc[0])
        rows.append(m)
    # Overall per product
    for et_product, grp in paired_df.groupby("et_product"):
        valid = grp[[obs_col, mod_col]].dropna()
        m = compute_metrics(valid[obs_col].values, valid[mod_col].values)
        m.update(site_id="ALL", et_product=et_product, igbp="")
        rows.append(m)
    cols = [
        "site_id",
        "igbp",
        "et_product",
        "N",
        "bias",
        "MAE",
        "RMSE",
        "r",
        "r2",
        "NSE",
    ]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Step 7 – Figures
# ---------------------------------------------------------------------------

_SITE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _stats_text(m: dict, fs: int = 8) -> str:
    return (
        f"N={m['N']}\nbias={m['bias']:+.1f}\n"
        f"RMSE={m['RMSE']:.1f}\nr2={m['r2']:.2f}\nNSE={m['NSE']:.2f}"
    )


def _sym_lim(obs, mod, pad_frac=0.05):
    lo = min(obs.min(), mod.min())
    hi = max(obs.max(), mod.max())
    pad = (hi - lo) * pad_frac if hi > lo else 50
    return lo - pad, hi + pad


def plot_scatter_per_site(
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "LE_obs",
    obs_label: str = "LE observed (W/m2)",
) -> None:
    """One scatter panel per (site x product) combination."""
    products = paired_df["et_product"].unique()
    site_ids = paired_df["site_id"].unique()
    n_rows, n_cols = len(site_ids), len(products)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for r, site_id in enumerate(site_ids):
        for c, prod in enumerate(products):
            ax = axes[r][c]
            grp = paired_df[
                (paired_df["site_id"] == site_id) & (paired_df["et_product"] == prod)
            ]
            if grp.empty:
                ax.set_visible(False)
                continue
            valid = grp[[obs_col, "LE_mod"]].dropna()
            if valid.empty:
                ax.set_visible(False)
                continue
            obs = valid[obs_col].values
            mod = valid["LE_mod"].values
            igbp = grp["igbp"].iloc[0]
            m_row = metrics_df[
                (metrics_df["site_id"] == site_id) & (metrics_df["et_product"] == prod)
            ]
            lim = _sym_lim(obs, mod)
            ax.scatter(obs, mod, s=28, alpha=0.7, edgecolors="none")
            ax.plot(lim, lim, "k--", lw=1)
            if len(obs) >= 2:
                sl, ic, *_ = stats.linregress(obs, mod)
                ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(obs_label)
            ax.set_ylabel("LE modelled (W/m2)")
            ax.set_title(f"{site_id} [{igbp}]  -  {prod}", fontsize=9)
            if not m_row.empty:
                ax.text(
                    0.03,
                    0.97,
                    _stats_text(m_row.iloc[0].to_dict()),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    fig.suptitle("TSEB-PT vs ICOS LE (W/m2)", fontsize=12, fontweight="bold")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Scatter per site -> {outfile}")


def plot_scatter_all_sites(
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "LE_obs",
    obs_label: str = "LE observed (W/m2)",
) -> None:
    """One panel per ET product, all sites coloured differently."""
    products = paired_df["et_product"].unique()
    n_cols = len(products)
    site_ids = paired_df["site_id"].unique()
    colors = {s: _SITE_COLORS[i % len(_SITE_COLORS)] for i, s in enumerate(site_ids)}

    fig, axes = plt.subplots(
        1, n_cols, figsize=(5.5 * n_cols, 5.5), constrained_layout=True, squeeze=False
    )

    for c, prod in enumerate(products):
        ax = axes[0][c]
        grp_prod = paired_df[paired_df["et_product"] == prod]
        for site_id in site_ids:
            grp = grp_prod[grp_prod["site_id"] == site_id]
            if grp.empty:
                continue
            valid = grp[[obs_col, "LE_mod"]].dropna()
            if valid.empty:
                continue
            ax.scatter(
                valid[obs_col],
                valid["LE_mod"],
                s=25,
                alpha=0.7,
                edgecolors="none",
                color=colors[site_id],
                label=f'{site_id} [{grp["igbp"].iloc[0]}]',
            )

        valid_all = grp_prod[[obs_col, "LE_mod"]].dropna()
        obs_all = valid_all[obs_col].values
        mod_all = valid_all["LE_mod"].values
        if len(obs_all) < 2:
            continue
        lim = _sym_lim(obs_all, mod_all)
        ax.plot(lim, lim, "k--", lw=1)
        sl, ic, *_ = stats.linregress(obs_all, mod_all)
        ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.5)

        m_all = metrics_df[
            (metrics_df["site_id"] == "ALL") & (metrics_df["et_product"] == prod)
        ]
        if not m_all.empty:
            ax.text(
                0.03,
                0.97,
                _stats_text(m_all.iloc[0].to_dict(), fs=9),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(obs_label, fontsize=11)
        ax.set_ylabel("LE modelled (W/m2)", fontsize=11)
        ax.set_title(f"All sites - {prod}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Scatter all sites -> {outfile}")


def plot_timeseries_per_site(
    paired_df: pd.DataFrame, outfile: Path, include_br_obs: bool = True
) -> None:
    """Observed vs modelled LE time series, one row per site."""
    site_ids = paired_df["site_id"].unique()
    n_rows = len(site_ids)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 3.5 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    prod_styles = {
        "007_et": ("tomato", "o-", "TSEB-PT 20 m"),
        "008_lstm-et": ("darkorange", "s--", "TSEB-PT 30 m LSTM"),
    }

    for ax, site_id in zip(axes, site_ids):
        grp_site = paired_df[paired_df["site_id"] == site_id].sort_values(
            "datetime_utc"
        )
        igbp = grp_site["igbp"].iloc[0]

        # Plot observed once (dedup by timestamp)
        obs_grp = grp_site.drop_duplicates("datetime_utc")
        ax.plot(
            obs_grp["datetime_utc"],
            obs_grp["LE_obs"],
            "o-",
            ms=4,
            color="steelblue",
            label="ICOS observed",
            zorder=3,
        )
        if include_br_obs and "LE_obs_BR" in obs_grp.columns:
            br_grp = obs_grp.dropna(subset=["LE_obs_BR"])
            if not br_grp.empty:
                ax.plot(
                    br_grp["datetime_utc"],
                    br_grp["LE_obs_BR"],
                    "d--",
                    ms=4,
                    color="mediumpurple",
                    label="ICOS observed (Bowen closure)",
                    zorder=2,
                )

        for prod, (col, ls, lbl) in prod_styles.items():
            sub = grp_site[grp_site["et_product"] == prod]
            if sub.empty:
                continue
            ax.plot(
                sub["datetime_utc"],
                sub["LE_mod"],
                ls,
                ms=4,
                color=col,
                label=lbl,
                alpha=0.85,
            )

        ax.set_ylabel("LE (W/m2)")
        ax.set_title(f"{site_id}  [{igbp}]", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("LE time series: TSEB-PT vs ICOS", fontsize=12, fontweight="bold")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Time series -> {outfile}")


def plot_metrics_bar(metrics_df: pd.DataFrame, outfile: Path) -> None:
    """Bar chart of RMSE, MAE, |bias| per site and product."""
    site_df = metrics_df[metrics_df["site_id"] != "ALL"].copy()
    products = site_df["et_product"].unique()
    n_prod = len(products)

    fig, axes = plt.subplots(
        1,
        n_prod,
        figsize=(max(6, 2.5 * len(site_df)) * n_prod, 4),
        constrained_layout=True,
        squeeze=False,
    )

    for c, prod in enumerate(products):
        ax = axes[0][c]
        sub = site_df[site_df["et_product"] == prod]
        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub["RMSE"], w, label="RMSE", color="tomato")
        ax.bar(x, sub["MAE"], w, label="MAE", color="steelblue")
        ax.bar(x + w, sub["bias"].abs(), w, label="|bias|", color="goldenrod")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["site_id"], rotation=30, ha="right")
        ax.set_ylabel("LE error (W/m2)")
        ax.set_title(f"Validation errors - {prod}", fontweight="bold")
        ax.legend()

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Metrics bar chart -> {outfile}")


def plot_scatter_per_site_et_br(
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "ET_obs_BR_mm_h",
    obs_label: str = "ET observed (Bowen closure) (mm/h)",
) -> None:
    """Scatter plot: Bowen-corrected ET per site x product."""
    products = paired_df["et_product"].unique()
    site_ids = paired_df["site_id"].unique()
    n_rows, n_cols = len(site_ids), len(products)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for r, site_id in enumerate(site_ids):
        for c, prod in enumerate(products):
            ax = axes[r][c]
            grp = paired_df[
                (paired_df["site_id"] == site_id) & (paired_df["et_product"] == prod)
            ]
            if grp.empty:
                ax.set_visible(False)
                continue
            valid = grp[[obs_col, "ET_mod_mm_h"]].dropna()
            if valid.empty:
                ax.set_visible(False)
                continue
            obs = valid[obs_col].values
            mod = valid["ET_mod_mm_h"].values
            igbp = grp["igbp"].iloc[0]
            m_row = metrics_df[
                (metrics_df["site_id"] == site_id) & (metrics_df["et_product"] == prod)
            ]
            lim = _sym_lim(obs, mod)
            ax.scatter(obs, mod, s=28, alpha=0.7, edgecolors="none")
            ax.plot(lim, lim, "k--", lw=1)
            if len(obs) >= 2:
                sl, ic, *_ = stats.linregress(obs, mod)
                ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(obs_label)
            ax.set_ylabel("ET modelled (mm/h)")
            ax.set_title(f"{site_id} [{igbp}]  -  {prod}", fontsize=9)
            if not m_row.empty:
                ax.text(
                    0.03,
                    0.97,
                    _stats_text(m_row.iloc[0].to_dict()),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    fig.suptitle(
        "TSEB-PT vs ICOS ET (mm/h) - Bowen Corrected", fontsize=12, fontweight="bold"
    )
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Scatter per site (Bowen-corrected ET) -> {outfile}")


def plot_scatter_all_sites_et_br(
    paired_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "ET_obs_BR_mm_h",
    obs_label: str = "ET observed (Bowen closure) (mm/h)",
) -> None:
    """Scatter plot: Bowen-corrected ET all sites, coloured by site."""
    products = paired_df["et_product"].unique()
    n_cols = len(products)
    site_ids = paired_df["site_id"].unique()
    colors = {s: _SITE_COLORS[i % len(_SITE_COLORS)] for i, s in enumerate(site_ids)}

    fig, axes = plt.subplots(
        1, n_cols, figsize=(5.5 * n_cols, 5.5), constrained_layout=True, squeeze=False
    )

    for c, prod in enumerate(products):
        ax = axes[0][c]
        grp_prod = paired_df[paired_df["et_product"] == prod]
        for site_id in site_ids:
            grp = grp_prod[grp_prod["site_id"] == site_id]
            if grp.empty:
                continue
            valid = grp[[obs_col, "ET_mod_mm_h"]].dropna()
            if valid.empty:
                continue
            ax.scatter(
                valid[obs_col],
                valid["ET_mod_mm_h"],
                s=25,
                alpha=0.7,
                edgecolors="none",
                color=colors[site_id],
                label=f'{site_id} [{grp["igbp"].iloc[0]}]',
            )

        valid_all = grp_prod[[obs_col, "ET_mod_mm_h"]].dropna()
        obs_all = valid_all[obs_col].values
        mod_all = valid_all["ET_mod_mm_h"].values
        if len(obs_all) < 2:
            continue
        lim = _sym_lim(obs_all, mod_all)
        ax.plot(lim, lim, "k--", lw=1)
        sl, ic, *_ = stats.linregress(obs_all, mod_all)
        ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.5)

        m_all = metrics_df[
            (metrics_df["site_id"] == "ALL") & (metrics_df["et_product"] == prod)
        ]
        if not m_all.empty:
            ax.text(
                0.03,
                0.97,
                _stats_text(m_all.iloc[0].to_dict(), fs=9),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(obs_label, fontsize=11)
        ax.set_ylabel("ET modelled (mm/h)", fontsize=11)
        ax.set_title(f"All sites - {prod}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Scatter all sites (Bowen-corrected ET) -> {outfile}")


def plot_timeseries_per_site_et_br(paired_df: pd.DataFrame, outfile: Path) -> None:
    """Bowen-corrected ET time series, one row per site."""
    site_ids = paired_df["site_id"].unique()
    n_rows = len(site_ids)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 3.5 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    prod_styles = {
        "007_et": ("tomato", "o-", "TSEB-PT 20 m"),
        "008_lstm-et": ("darkorange", "s--", "TSEB-PT 30 m LSTM"),
    }

    for ax, site_id in zip(axes, site_ids):
        grp_site = paired_df[paired_df["site_id"] == site_id].sort_values(
            "datetime_utc"
        )
        igbp = grp_site["igbp"].iloc[0]

        # Plot observed Bowen-corrected ET once (dedup by timestamp)
        obs_grp = grp_site.drop_duplicates("datetime_utc")
        br_grp = obs_grp.dropna(subset=["ET_obs_BR_mm_h"])
        if not br_grp.empty:
            ax.plot(
                br_grp["datetime_utc"],
                br_grp["ET_obs_BR_mm_h"],
                "d-",
                ms=5,
                color="mediumpurple",
                label="ICOS observed (Bowen closure)",
                zorder=3,
            )

        # Plot modelled ET
        for prod, (col, ls, lbl) in prod_styles.items():
            sub = grp_site[grp_site["et_product"] == prod]
            if sub.empty:
                continue
            valid_et = sub.dropna(subset=["ET_mod_mm_h"])
            if valid_et.empty:
                continue
            ax.plot(
                valid_et["datetime_utc"],
                valid_et["ET_mod_mm_h"],
                ls,
                ms=4,
                color=col,
                label=lbl,
                alpha=0.85,
            )

        ax.set_ylabel("ET (mm/h)")
        ax.set_title(f"{site_id} [{igbp}] - Bowen Corrected ET", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "ET time series (Bowen closure): TSEB-PT vs ICOS",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Time series (Bowen-corrected ET) -> {outfile}")


def plot_daily_et_scatter(
    daily_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "ET_obs_BR_mm_day",
    obs_label: str = "ET observed (Bowen closure) (mm/day)",
) -> None:
    """Scatter plot: daily ET per site x product."""
    if daily_df.empty:
        logger.warning(f"Skipping daily ET scatter: no data for {obs_col}")
        return

    products = daily_df["et_product"].unique()
    site_ids = daily_df["site_id"].unique()
    n_rows, n_cols = len(site_ids), len(products)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for r, site_id in enumerate(site_ids):
        for c, prod in enumerate(products):
            ax = axes[r][c]
            grp = daily_df[
                (daily_df["site_id"] == site_id) & (daily_df["et_product"] == prod)
            ]
            if grp.empty:
                ax.set_visible(False)
                continue
            valid = grp[[obs_col, "ET_mod_mm_day"]].dropna()
            if valid.empty:
                ax.set_visible(False)
                continue

            obs = valid[obs_col].values
            mod = valid["ET_mod_mm_day"].values
            igbp = grp["igbp"].iloc[0]

            lim = _sym_lim(obs, mod)
            ax.scatter(obs, mod, s=35, alpha=0.7, edgecolors="none")
            ax.plot(lim, lim, "k--", lw=1)

            if len(obs) >= 2:
                sl, ic, *_ = stats.linregress(obs, mod)
                ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.2)
                m = compute_metrics(obs, mod)
                ax.text(
                    0.03,
                    0.97,
                    _stats_text(m),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(obs_label, fontsize=10)
            ax.set_ylabel("ET modelled (mm/day)", fontsize=10)
            ax.set_title(f"{site_id} [{igbp}]  -  {prod}", fontsize=9)

    fig.suptitle("Daily ET: TSEB-PT vs ICOS", fontsize=12, fontweight="bold")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Daily ET scatter plot -> {outfile}")


def plot_daily_et_all_sites(
    daily_df: pd.DataFrame,
    outfile: Path,
    obs_col: str = "ET_obs_BR_mm_day",
    obs_label: str = "ET observed (Bowen closure) (mm/day)",
) -> None:
    """Scatter plot: daily ET all sites, coloured by site."""
    if daily_df.empty:
        logger.warning(f"Skipping daily ET all-sites scatter: no data for {obs_col}")
        return

    products = daily_df["et_product"].unique()
    n_cols = len(products)
    site_ids = daily_df["site_id"].unique()
    colors = {s: _SITE_COLORS[i % len(_SITE_COLORS)] for i, s in enumerate(site_ids)}

    fig, axes = plt.subplots(
        1, n_cols, figsize=(5.5 * n_cols, 5.5), constrained_layout=True, squeeze=False
    )

    for c, prod in enumerate(products):
        ax = axes[0][c]
        grp_prod = daily_df[daily_df["et_product"] == prod]

        for site_id in site_ids:
            grp = grp_prod[grp_prod["site_id"] == site_id]
            if grp.empty:
                continue
            valid = grp[[obs_col, "ET_mod_mm_day"]].dropna()
            if valid.empty:
                continue
            ax.scatter(
                valid[obs_col],
                valid["ET_mod_mm_day"],
                s=30,
                alpha=0.7,
                edgecolors="none",
                color=colors[site_id],
                label=f'{site_id} [{grp["igbp"].iloc[0]}]',
            )

        valid_all = grp_prod[[obs_col, "ET_mod_mm_day"]].dropna()
        obs_all = valid_all[obs_col].values
        mod_all = valid_all["ET_mod_mm_day"].values

        if len(obs_all) >= 2:
            lim = _sym_lim(obs_all, mod_all)
            ax.plot(lim, lim, "k--", lw=1)
            sl, ic, *_ = stats.linregress(obs_all, mod_all)
            ax.plot(lim, [sl * x + ic for x in lim], "r-", lw=1.5)

            m = compute_metrics(obs_all, mod_all)
            ax.text(
                0.03,
                0.97,
                _stats_text(m, fs=9),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
            ax.set_xlim(lim)
            ax.set_ylim(lim)

        ax.set_xlabel(obs_label, fontsize=11)
        ax.set_ylabel("ET modelled (mm/day)", fontsize=11)
        ax.set_title(f"All sites - {prod}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Daily ET all-sites scatter -> {outfile}")


def plot_daily_et_timeseries(
    daily_df: pd.DataFrame, outfile: Path, obs_col: str = "ET_obs_BR_mm_day"
) -> None:
    """Time series of daily ET, one row per site."""
    if daily_df.empty:
        logger.warning("Skipping daily ET time series: no data")
        return

    site_ids = daily_df["site_id"].unique()
    n_rows = len(site_ids)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 3.0 * n_rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    prod_styles = {
        "007_et": ("tomato", "o-", "TSEB-PT 20 m"),
        "008_lstm-et": ("darkorange", "s--", "TSEB-PT 30 m LSTM"),
    }

    for ax, site_id in zip(axes, site_ids):
        grp_site = daily_df[daily_df["site_id"] == site_id].sort_values("date")
        igbp = grp_site["igbp"].iloc[0]

        # Plot observed daily ET
        obs_valid = grp_site.dropna(subset=[obs_col])
        if not obs_valid.empty:
            dates = pd.to_datetime(obs_valid["date"])
            ax.plot(
                dates,
                obs_valid[obs_col],
                "D-",
                ms=5,
                color="mediumpurple",
                label="ICOS observed (Bowen closure)",
                zorder=3,
            )

        # Plot modelled daily ET
        for prod, (col, ls, lbl) in prod_styles.items():
            sub = grp_site[grp_site["et_product"] == prod]
            if sub.empty:
                continue
            valid_et = sub.dropna(subset=["ET_mod_mm_day"])
            if not valid_et.empty:
                dates = pd.to_datetime(valid_et["date"])
                ax.plot(
                    dates,
                    valid_et["ET_mod_mm_day"],
                    ls,
                    ms=4,
                    color=col,
                    label=lbl,
                    alpha=0.85,
                )

        ax.set_ylabel("Daily ET (mm/day)")
        ax.set_title(f"{site_id} [{igbp}]", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Daily ET time series: TSEB-PT vs ICOS", fontsize=12, fontweight="bold"
    )
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Daily ET time series -> {outfile}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    icos_archive_dir: Path,
    output_root: Union[Path, List[Path]],
    output_dir: Path,
    et_subdirs=None,
    extraction_radius_m: float = 150.0,
    max_qc: int = 0,
    le_band_name: str = "LE1",
    min_daily_obs: int = 24,
    drop_negative_daily_obs: bool = True,
) -> pd.DataFrame:
    """Run the full TSEB-ICOS validation workflow.

    Parameters
    ----------
    icos_archive_dir:
        Folder with ICOS ZIP archives.
    output_root:
        Root of pyTSEB outputs, or list of roots, e.g.
        [/path/to/output/2024, /path/to/output/2025].
        The function automatically looks under
        output_root / <tile> / <et_subdir> for every S2 tile that
        contains an ICOS tower.  A clear warning is emitted for any
        tower whose tile directory does not yet exist (run
        run_sen-et.py for that tile first).
    output_dir:
        Where to save figures and CSV files.
    et_subdirs:
        Which ET product sub-directories to validate.
        Default: ['007_et', '008_lstm-et']
    extraction_radius_m:
        Spatial extraction radius around the tower (metres).
        150 m ~ 7x7 pixels at 20 m resolution.
    max_qc:
        Maximum LE_F_MDS_QC to accept (0 = measured only).
    le_band_name:
        Name of the instantaneous LE band in the .data/ folder.
    min_daily_obs:
        Minimum number of valid half-hourly LE records needed to compute
        observed ET_day for a date.
    drop_negative_daily_obs:
        If True, negative observed ET_day values are treated as NaN.

    Returns
    -------
    pd.DataFrame with the paired (obs, mod) records.
    """
    if et_subdirs is None:
        et_subdirs = ["007_et", "008_lstm-et"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== TSEB-PT / ICOS validation ===")
    logger.info(f"ICOS archive : {icos_archive_dir}")
    logger.info(f"Output root  : {output_root}")
    logger.info(f"ET products  : {et_subdirs}")
    logger.info(f"QC filter    : LE_F_MDS_QC <= {max_qc}")
    logger.info(f"Radius       : {extraction_radius_m} m")
    logger.info(f"Min daily obs: {min_daily_obs}")
    logger.info(f"Drop neg ETd : {drop_negative_daily_obs}")

    # 1. Load ICOS
    sites = load_icos_sites(icos_archive_dir, max_qc=max_qc)
    if not sites:
        logger.error("No ICOS sites loaded - aborting.")
        return pd.DataFrame()

    # 2. Load S2 grid
    s2_grid = _load_s2_grid()

    # 3-4. Match and extract
    paired_df = collect_paired_data(
        sites,
        output_root,
        et_subdirs,
        s2_grid,
        extraction_radius_m=extraction_radius_m,
        le_band_name=le_band_name,
        min_daily_obs=min_daily_obs,
        drop_negative_daily_obs=drop_negative_daily_obs,
    )

    if paired_df.empty:
        logger.error("No matched pairs - no output produced.")
        return paired_df

    # Save paired data
    paired_df["ET_mod_mm_h"] = _le_to_et_mm_h(paired_df["LE_mod"])
    paired_df["ET_obs_BR_mm_h"] = _le_to_et_mm_h(paired_df["LE_obs_BR"])

    csv_pairs = output_dir / "paired_le_obs_mod.csv"
    paired_df.to_csv(csv_pairs, index=False)
    logger.info(f"Paired data -> {csv_pairs}")

    # 5. Metrics (standard LE)
    metrics_df = compute_all_metrics(paired_df, obs_col="LE_obs")
    csv_metrics = output_dir / "metrics_table.csv"
    metrics_df.to_csv(csv_metrics, index=False, float_format="%.3f")
    logger.info(f"Metrics table -> {csv_metrics}")
    logger.info("\n" + metrics_df.to_string(index=False))

    # 5b. Metrics (Bowen-ratio closure LE)
    br_count = (
        paired_df["LE_obs_BR"].notna().sum() if "LE_obs_BR" in paired_df.columns else 0
    )
    if br_count > 0:
        metrics_br_df = compute_all_metrics(paired_df, obs_col="LE_obs_BR")
        csv_metrics_br = output_dir / "metrics_table_bowen_ratio.csv"
        metrics_br_df.to_csv(csv_metrics_br, index=False, float_format="%.3f")
        logger.info(f"Bowen-ratio metrics table -> {csv_metrics_br}")
        logger.info("\n" + metrics_br_df.to_string(index=False))
    else:
        metrics_br_df = pd.DataFrame()
        logger.warning(
            "No valid Bowen-ratio closure records found; skipping BR metrics."
        )

    # 5c. ET metrics using Bowen-corrected observed LE vs modelled LE
    et_br_count = paired_df[["ET_obs_BR_mm_h", "ET_mod_mm_h"]].dropna().shape[0]
    if et_br_count > 0:
        metrics_et_br_df = compute_all_metrics(
            paired_df, obs_col="ET_obs_BR_mm_h", mod_col="ET_mod_mm_h"
        )
        csv_metrics_et_br = output_dir / "metrics_table_bowen_ratio_et_mm_h.csv"
        metrics_et_br_df.to_csv(csv_metrics_et_br, index=False, float_format="%.6f")
        logger.info(f"Bowen-ratio ET metrics table -> {csv_metrics_et_br}")
        logger.info("\n" + metrics_et_br_df.to_string(index=False))
    else:
        metrics_et_br_df = pd.DataFrame()
        logger.warning("No valid Bowen-ratio ET records found; skipping BR ET metrics.")

    # 6. Figures
    plot_scatter_per_site(
        paired_df,
        metrics_df,
        output_dir / "scatter_per_site.png",
        obs_col="LE_obs",
        obs_label="LE observed (W/m2)",
    )
    plot_scatter_all_sites(
        paired_df,
        metrics_df,
        output_dir / "scatter_all_sites.png",
        obs_col="LE_obs",
        obs_label="LE observed (W/m2)",
    )

    if not metrics_br_df.empty:
        plot_scatter_per_site(
            paired_df,
            metrics_br_df,
            output_dir / "scatter_per_site_bowen_ratio.png",
            obs_col="LE_obs_BR",
            obs_label="LE observed (Bowen closure) (W/m2)",
        )
        plot_scatter_all_sites(
            paired_df,
            metrics_br_df,
            output_dir / "scatter_all_sites_bowen_ratio.png",
            obs_col="LE_obs_BR",
            obs_label="LE observed (Bowen closure) (W/m2)",
        )

    if not metrics_et_br_df.empty:
        plot_scatter_per_site_et_br(
            paired_df,
            metrics_et_br_df,
            output_dir / "scatter_per_site_bowen_corrected_et.png",
            obs_col="ET_obs_BR_mm_h",
            obs_label="ET observed (Bowen closure) (mm/h)",
        )
        plot_scatter_all_sites_et_br(
            paired_df,
            metrics_et_br_df,
            output_dir / "scatter_all_sites_bowen_corrected_et.png",
            obs_col="ET_obs_BR_mm_h",
            obs_label="ET observed (Bowen closure) (mm/h)",
        )
        plot_timeseries_per_site_et_br(
            paired_df, output_dir / "timeseries_per_site_bowen_corrected_et.png"
        )

    plot_timeseries_per_site(
        paired_df, output_dir / "timeseries_per_site.png", include_br_obs=True
    )
    plot_metrics_bar(metrics_df, output_dir / "metrics_bar.png")

    if not metrics_br_df.empty:
        plot_metrics_bar(metrics_br_df, output_dir / "metrics_bar_bowen_ratio.png")

    if not metrics_et_br_df.empty:
        plot_metrics_bar(
            metrics_et_br_df, output_dir / "metrics_bar_bowen_corrected_et.png"
        )

    # Daily ET visualization (if aggregated records exist)
    daily_df = paired_df.attrs.get("daily_df", pd.DataFrame())
    if not daily_df.empty:
        # Compute daily ET metrics (Bowen-ratio closure)
        daily_et_br_count = (
            daily_df[["ET_obs_BR_mm_day", "ET_mod_mm_day"]].dropna().shape[0]
        )
        if daily_et_br_count > 0:
            daily_metrics_br_df = compute_all_metrics(
                daily_df, obs_col="ET_obs_BR_mm_day", mod_col="ET_mod_mm_day"
            )
            csv_daily_et_br = output_dir / "metrics_table_daily_et_bowen_ratio.csv"
            daily_metrics_br_df.to_csv(
                csv_daily_et_br, index=False, float_format="%.3f"
            )
            logger.info(f"Daily ET (Bowen ratio) metrics -> {csv_daily_et_br}")
            logger.info("\n" + daily_metrics_br_df.to_string(index=False))

            # Daily ET plots
            plot_daily_et_scatter(
                daily_df,
                daily_metrics_br_df,
                output_dir / "scatter_daily_et.png",
                obs_col="ET_obs_BR_mm_day",
                obs_label="ET observed (Bowen closure) (mm/day)",
            )
            plot_daily_et_all_sites(
                daily_df,
                output_dir / "scatter_daily_et_all_sites.png",
                obs_col="ET_obs_BR_mm_day",
                obs_label="ET observed (Bowen closure) (mm/day)",
            )
            plot_daily_et_timeseries(
                daily_df,
                output_dir / "timeseries_daily_et.png",
                obs_col="ET_obs_BR_mm_day",
            )
        else:
            logger.warning(
                "No valid daily ET records (Bowen ratio) found; skipping daily ET plots."
            )
    else:
        logger.warning("No daily ET data aggregated; skipping daily ET analysis.")

    logger.info("=== Validation complete ===")
    return paired_df


# ---------------------------------------------------------------------------
# Settings - edit here and run `python validate_tseb_icos.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Folder containing ICOS ZIP archives
    ICOS_ARCHIVE_DIR = Path("/path/to/ICOS/ETC_L2_ARCHIVE")

    # Root of the pyTSEB output tree (same as output_dir in run_sen-et.py)
    # The script automatically locates the correct S2 tile for every ICOS
    # tower using the bundled S2 grid GeoJSON and scans that tile's output
    # directory.  Run run_sen-et.py for any missing tiles first.
    OUTPUT_ROOT = [
        Path("/path/to/output/2024"),
        Path("/path/to/output/2025"),
    ]

    # Which ET product sub-directories to validate.
    # Remove '008_lstm-et' if generate_lstm_like was not run.
    ET_SUBDIRS = ["007_et", "008_lstm-et"]

    # Base directory for validation output (will create subdirs for each radius)
    VALIDATION_OUTPUT_BASE = Path("/path/to/output/validation_2024_2025")

    # List of circular extraction radii to test (metres).
    # Set to a single value [20.0] for a single run, or multiple values
    # [10.0, 15.0, 20.0, 25.0] for a radius sensitivity analysis.
    EXTRACTION_RADII_M = [20.0, 30, 40, 50, 60, 70]

    # 0 = directly measured only; 1 = also keep good-quality gap-fills
    MAX_QC = 1

    # Minimum number of valid half-hourly LE points required per day for
    # observed daily ET. 48 means full-day coverage; 24 is a conservative
    # half-day minimum.
    MIN_DAILY_OBS = 24

    # Drop negative observed daily ET values (often condensation / closure
    # artefacts when daily coverage is sparse).
    DROP_NEGATIVE_DAILY_OBS = True

    # Run validation for each extraction radius
    for extraction_radius_m in EXTRACTION_RADII_M:
        # Create subdirectory for this radius
        output_dir = VALIDATION_OUTPUT_BASE / f"radius_{extraction_radius_m:.1f}m"

        logger.info(f'\n{"="*70}')
        logger.info(
            f"Running validation with extraction radius = {extraction_radius_m} m"
        )
        logger.info(f"Output directory: {output_dir}")
        logger.info(f'{"="*70}\n')

        main(
            icos_archive_dir=ICOS_ARCHIVE_DIR,
            output_root=OUTPUT_ROOT,
            output_dir=output_dir,
            et_subdirs=ET_SUBDIRS,
            extraction_radius_m=extraction_radius_m,
            max_qc=MAX_QC,
            min_daily_obs=MIN_DAILY_OBS,
            drop_negative_daily_obs=DROP_NEGATIVE_DAILY_OBS,
        )

    logger.info(f'\n{"="*70}')
    logger.info("All extraction radius tests completed!")
    logger.info(f"Results saved in subdirectories of: {VALIDATION_OUTPUT_BASE}")
    logger.info(f'{"="*70}\n')
