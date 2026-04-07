"""
TSEB (Two-Source Energy Balance) module.

High-level orchestration for computing evapotranspiration using the
TSEB-PT model from pyTSEB. Handles ERA5 meteorological input preparation
and model execution.
"""

import datetime as dt
import gc
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from loguru import logger
from osgeo import gdal
from pyTSEB import TSEB as _tseb_core
from pyTSEB import resistances as _tseb_res
from pyTSEB.meteo_utils import calc_sun_angles
from pyTSEB.PyTSEB import PyTSEB, S_P, S_A

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for server runs
import matplotlib.pyplot as plt

from sen_et_openeo.era5 import ERA5TimeSeriesProcessor, get_default_rsi_meta
from sen_et_openeo.utils.geoloader import _getECMWFIntegratedData
from sen_et_openeo.utils.meteo import calc_lapse_rate_moist, Z_BH, GRAVITY
from sen_et_openeo.utils.warping import warp_in_memory


# ---------------------------------------------------------------------------
# WorldCover land-cover look-up table
# Source: DHI Sen-ET-OpenEO-toolbox / WorldCover10m_2020_LUT.csv
# Keys are ESA WorldCover 10m class values.
# 'vh'            : default canopy height (m)
# 'is_herbaceous' : whether LAI-based height scaling applies
# ---------------------------------------------------------------------------
_WORLDCOVER_LUT = {
    10:  {'vh': 10.0, 'is_herbaceous': False},   # Tree cover
    20:  {'vh':  2.0, 'is_herbaceous': False},   # Shrubland
    30:  {'vh':  1.0, 'is_herbaceous': True},    # Grassland
    40:  {'vh':  1.5, 'is_herbaceous': True},    # Cropland
    50:  {'vh': 10.0, 'is_herbaceous': False},   # Built-up
    60:  {'vh':  0.1, 'is_herbaceous': False},   # Bare/Sparse Vegetation
    70:  {'vh':  0.1, 'is_herbaceous': False},   # Snow and Ice
    80:  {'vh':  0.1, 'is_herbaceous': False},   # Permanent water bodies
    90:  {'vh':  1.0, 'is_herbaceous': True},    # Herbaceous wetland
    95:  {'vh':  5.0, 'is_herbaceous': False},   # Mangroves
    100: {'vh':  0.3, 'is_herbaceous': False},   # Moss and lichen
    0:   {'vh':  0.1, 'is_herbaceous': False},   # No data
}

# ---------------------------------------------------------------------------
# Mapping from ESA WorldCover 10m class values to pyTSEB landcover codes.
# pyTSEB codes (from pyTSEB.ResistanceCanopyLayer):
#   WATER=0, CONIFER_E=1, BROADLEAVED_E=2, CONIFER_D=3, BROADLEAVED_D=4,
#   FOREST_MIXED=5, SHRUB_C=6, SHRUB_O=7, SAVANNA_WOODY=8, SAVANNA=9,
#   GRASS=10, WETLAND=11, CROP=12, URBAN=13, CROP_MOSAIC=14, SNOW=15,
#   BARREN=16
# ---------------------------------------------------------------------------
_WORLDCOVER_TO_PYTSEB = {
    0:   0,   # No data       → WATER  (masked by input_mask)
    10:  2,   # Tree cover    → BROADLEAVED_E
    20:  7,   # Shrubland     → SHRUB_O
    30:  10,  # Grassland     → GRASS
    40:  12,  # Cropland      → CROP
    50:  13,  # Built-up      → URBAN  (masked by input_mask)
    60:  16,  # Bare/sparse   → BARREN
    70:  15,  # Snow and ice  → SNOW   (masked by input_mask)
    80:  0,   # Water bodies  → WATER  (masked by input_mask)
    90:  11,  # Wetland       → WETLAND
    95:  2,   # Mangroves     → BROADLEAVED_E
    100: 10,  # Moss/lichen   → GRASS
}


def remap_worldcover_for_pytseb(worldcover_file, outfile):
    """Remap ESA WorldCover class values to pyTSEB internal landcover codes.

    The result is cached on disk — if ``outfile`` already exists it is
    returned immediately without recomputing.

    Args:
        worldcover_file (Path): WorldCover GeoTIFF (uint8, ESA class values).
        outfile (Path): Destination path for the remapped GeoTIFF.

    Returns:
        Path: Path to the written (or already existing) GeoTIFF.
    """
    outfile = Path(outfile)
    if outfile.exists():
        return outfile
    with rasterio.open(worldcover_file) as src:
        wc = src.read(1).astype(np.uint8)
        profile = src.profile.copy()
    mapped = np.zeros_like(wc, dtype=np.uint8)
    for wc_class, tseb_class in _WORLDCOVER_TO_PYTSEB.items():
        mapped[wc == wc_class] = tseb_class
    profile.update(dtype=rasterio.uint8, nodata=255)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(outfile, 'w', **profile) as dst:
        dst.write(mapped, 1)
    logger.debug(f'WorldCover remapped to pyTSEB classes: {outfile}')
    return outfile


def calc_fg(fapar_file, lai_file, doy, outfile,
            min_frac_green=0.01, n_iter=50):
    """
    Compute fraction of green vegetation (F_G) from FAPAR, LAI and solar
    zenith angle (SZA) at solar noon, using an iterative solver.

    SZA is computed analytically at the tile centre (adequate for a single
    Sentinel-2 tile, ~110 km).  Follows the ``calc_fg`` approach in the
    DHI Sen-ET-OpenEO-toolbox using ``TSEB.calc_F_theta_campbell``.

    Args:
        fapar_file (Path): FAPAR GeoTIFF on the S2 20 m grid.
        lai_file   (Path): LAI GeoTIFF on the S2 20 m grid.
        doy        (int) : Day of year of the S2 composite.
        outfile    (Path): Output F_G GeoTIFF path.
        min_frac_green (float): Minimum allowed F_G. Default 0.01.
        n_iter     (int) : Maximum iterations. Default 50.

    Returns:
        Path: Path to the written F_G GeoTIFF.
    """
    with rasterio.open(fapar_file) as src:
        fapar     = src.read(1).astype(np.float32)
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        h, w      = src.height, src.width

    with rasterio.open(lai_file) as src:
        lai = src.read(1).astype(np.float32)
    # Cap LAI to the physical maximum accepted by the iterative solver;
    # outliers in the BIOPAR UDP (up to ~14 m²/m²) saturate FIPAR→1
    # and collapse f_g to FAPAR, producing unreliable fractions.
    _LAI_MAX = 8.0
    lai = np.clip(lai, 0.0, _LAI_MAX)

    # SZA at solar noon at the tile centre (scalar)
    cx, cy = rasterio.transform.xy(transform, h // 2, w // 2)
    to_wgs84 = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lon_c, lat_c = to_wgs84.transform(cx, cy)
    sza, _ = calc_sun_angles(lat_c, lon_c, 0.0, doy, 12.0)

    f_g = np.ones(lai.shape, dtype=np.float32)
    converged = np.zeros(lai.shape, dtype=bool)
    # Pixels with negligible LAI or FAPAR converge immediately
    converged[np.logical_or(lai <= 0.2, fapar <= 0.1)] = True

    for _ in range(n_iter):
        f_g_old = f_g.copy()
        mask = ~converged
        if not np.any(mask):
            break
        fg_m    = np.where(f_g[mask] > 0, f_g[mask], min_frac_green)
        lai_eff = lai[mask] / fg_m
        fipar   = _tseb_core.calc_F_theta_campbell(
            sza, lai_eff, w_C=1, Omega0=1, x_LAD=1)
        f_g[mask] = fapar[mask] / fipar
        f_g = np.clip(f_g, min_frac_green, 1.0)
        converged = np.logical_or(
            np.isnan(f_g),
            np.abs(f_g - f_g_old) < 0.02,
        )

    outfile = Path(outfile)
    profile.update(count=1, dtype=rasterio.float32, nodata=-9999.0)
    with rasterio.open(outfile, 'w', **profile) as dst:
        dst.write(
            np.where(np.isfinite(f_g), f_g, -9999.0).astype(np.float32), 1)
    logger.debug(f'F_G written to {outfile}')
    return outfile


def calc_canopy_height(lai_file, worldcover_file, fg_file, outfile):
    """
    Estimate canopy height (H_C, m) from LAI, ESA WorldCover land-cover
    class and fraction green (F_G).

    For herbaceous classes (grassland, cropland, wetland) the height is
    scaled with the plant area index (PAI = LAI / F_G) following the
    DHI Sen-ET-OpenEO-toolbox approach.

    Args:
        lai_file        (Path): LAI GeoTIFF on the S2 20 m grid.
        worldcover_file (Path): WorldCover GeoTIFF warped to the S2 20 m grid.
        fg_file         (Path): F_G GeoTIFF (output of :func:`calc_fg`).
        outfile         (Path): Output H_C GeoTIFF path.

    Returns:
        Path: Path to the written H_C GeoTIFF.
    """
    with rasterio.open(lai_file) as src:
        lai     = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    # Apply the same LAI cap used in calc_fg and the TSEB block loop
    # so that herbaceous canopy heights are not over-estimated from outliers.
    _LAI_MAX = 8.0
    lai = np.clip(lai, 0.0, _LAI_MAX)

    with rasterio.open(worldcover_file) as src:
        landcover = src.read(1).astype(np.int32)

    with rasterio.open(fg_file) as src:
        fg = src.read(1).astype(np.float32)

    fg = np.where(fg > 0, fg, 0.01)   # guard against zero-division
    h_c = np.full(lai.shape, np.nan, dtype=np.float32)

    # Round to nearest WorldCover decade; preserve class 95 (Mangroves)
    lc_rounded = np.where(landcover == 95, 95, 10 * (landcover // 10))

    for lc_class, entry in _WORLDCOVER_LUT.items():
        vh   = entry['vh']
        mask = lc_rounded == lc_class
        if not np.any(mask):
            continue
        h_c[mask] = vh
        if entry['is_herbaceous']:
            pai = (lai / fg)[mask]
            h_c[mask] = (0.1 * vh
                         + 0.9 * vh * np.minimum((pai / vh) ** 3.0, 1.0))

    outfile = Path(outfile)
    profile.update(count=1, dtype=rasterio.float32, nodata=-9999.0)
    with rasterio.open(outfile, 'w', **profile) as dst:
        dst.write(
            np.where(np.isfinite(h_c), h_c, -9999.0).astype(np.float32), 1)
    logger.debug(f'H_C written to {outfile}')
    return outfile


def compute_meteo_for_tseb(time, elev_file, time_zone, era5col, outdir):
    """
    Compute pyTSEB meteorological inputs from ERA5 data at the S3 overpass
    time. ERA5 bands are temporally interpolated to the overpass time and
    spatially resampled to the 20m S2 grid using the elevation file as
    reference.

    Args:
        time (datetime): S3 overpass datetime.
        elev_file (Path): Path to the DEM GeoTIFF (used as spatial reference).
        time_zone (int): UTC offset in whole hours (e.g. 0 for UTC, 2 for
            UTC+2).
        era5col (ERA5Collection): ERA5 collection covering the required dates.
        outdir (Path): Directory to write the output GeoTIFF files to.

    Returns:
        dict: Mapping of pyTSEB parameter names to output GeoTIFF Paths:
            - ``T_A1``    : air temperature at 100 m blending height (K)
            - ``u``       : wind speed at 100 m (m/s)
            - ``ea``      : vapour pressure (mb)
            - ``p``       : air pressure (mb)
            - ``S_dn``    : instantaneous shortwave radiation at overpass (W/m²)
            - ``S_dn_24`` : daily mean shortwave radiation (W/m²)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    timestr    = time.strftime('%Y%m%dT%H%M%S')
    datestr    = time.strftime('%Y%m%d')
    datestr_nc = time.strftime('%Y-%m-%d')

    # Reference raster profile and elevation data from DEM file
    with rasterio.open(elev_file) as src:
        ref_profile = src.profile.copy()
        elev_data = src.read(1).astype(np.float32)
    ref_profile.update(count=1, dtype=rasterio.float32, nodata=-9999.0)

    # Load ERA5 bands temporally interpolated to overpass time
    # and resampled to the 20m S2 grid.
    # Note: ssrd (solar radiation) is an accumulated field — the
    # ERA5TimeSeriesProcessor reads it as a 24h daily integral via
    # _getECMWFIntegratedData, giving a daily mean in W/m².
    meteo_settings = {
        'bands': ['t2m', 'z', 'd2m', 'sp', 'u100', 'v100', 'ssrd']
    }
    meteo_rsi_meta = get_default_rsi_meta().get('ERA5')

    meteo_ts = ERA5TimeSeriesProcessor(
        [time],
        None,
        elev_file,
        time_zone,
        era5col,
        meteo_settings,
        rsi_meta=meteo_rsi_meta,
    ).compute_ts()

    def get_band(name):
        idx = meteo_ts.bands.index(name)
        return np.squeeze(meteo_ts.data[idx, 0, ...]).astype(np.float32)

    t2m      = get_band('t2m')   # K  — 2m air temperature
    z_geopot = get_band('z')     # m²/s² — geopotential
    d2m      = get_band('d2m')   # K  — 2m dewpoint temperature
    sp       = get_band('sp')    # Pa — surface pressure
    u100     = get_band('u100')  # m/s — u-component of wind at 100m
    v100     = get_band('v100')  # m/s — v-component of wind at 100m
    ssrd_24h = get_band('ssrd')  # W/m² — daily mean solar radiation

    # --- Derived meteorological quantities ---
    z_m  = z_geopot / GRAVITY   # geopotential → geometric height (m)
    # Vapour pressure from dewpoint via Magnus formula (mb)
    ea   = 6.1078 * np.exp(17.269 * (d2m - 273.15) / (235.5 + (d2m - 273.15)))
    p_mb = sp / 100.0            # Pa → mb
    ws   = np.sqrt(u100**2 + v100**2)  # scalar wind speed (m/s)

    # Air temperature at 100m blending height above surface:
    # 1. Extrapolate from 2m to 0m datum using geopotential height as ref.
    # 2. Re-apply lapse rate from 0m datum up to (DEM elev + 100m).
    lapse   = calc_lapse_rate_moist(t2m, ea, p_mb)
    t_datum = t2m - lapse * (0.0 - (z_m + 2.0))   # T at 0m datum
    t_a     = t_datum - lapse * (elev_data + Z_BH)  # T at blending height

    def write_tif(data, suffix):
        path = outdir / f'{suffix}.tif'
        data = np.squeeze(np.where(np.isfinite(data), data, -9999.0))
        prof = ref_profile.copy()
        prof.update(compress='deflate', PREDICTOR=2)
        with rasterio.open(path, 'w', **prof) as dst:
            dst.write(data.astype(np.float32), 1)
        return path

    paths = {
        'T_A1':    write_tif(t_a,      f'{timestr}_TA'),
        'u':       write_tif(ws,       f'{timestr}_WS'),
        'ea':      write_tif(ea,       f'{timestr}_EA'),
        'p':       write_tif(p_mb,     f'{timestr}_PA'),
        'S_dn_24': write_tif(ssrd_24h, f'{datestr}_SW-IN-DD'),
    }

    # Instantaneous solar radiation: 1-hour accumulation ending at
    # overpass time, converted to average W/m² over that hour.
    ncfile_path = str(
        era5col.df[era5col.df.day == datestr_nc].path.values[0])
    start_sdn = time - dt.timedelta(hours=1)
    s_dn_raw, gt, proj = _getECMWFIntegratedData(
        ncfile_path, 'ssrd', start_sdn, time_window=1)
    s_dn = warp_in_memory(
        s_dn_raw, gt, proj, str(elev_file)).astype(np.float32)
    paths['S_dn'] = write_tif(s_dn, f'{timestr}_SW-IN')

    # Free all large full-tile arrays before the TSEB block loop.
    # Each float32 array is ~120 MB for a 5490x5490 S2 tile;
    # meteo_ts holds a 7-band stack (~840 MB).
    del (meteo_ts,
         t2m, z_geopot, d2m, sp, u100, v100, ssrd_24h,
         z_m, ea, p_mb, ws, lapse, t_datum, t_a, elev_data,
         s_dn_raw, s_dn)
    gc.collect()

    return paths


# ---------------------------------------------------------------------------
# Tiled TSEB execution
# ---------------------------------------------------------------------------

def _process_tseb_tiled(model, output_file, block_size=1024, lst_datetime=None):
    """Block-wise TSEB execution that replaces ``PyTSEB.process_local_image()``.

    ``process_local_image`` reads every input raster for the full S2 tile into
    RAM simultaneously (≈20 float32 arrays × ~120 MB each = >2 GB peak).  This
    function reads and processes the image one ``block_size × block_size``
    window at a time, calling ``PyTSEB.run()`` per block and writing output
    GeoTIFFs incrementally. Memory usage is reduced to O(block_size²) instead
    of O(full_tile²).

    The output layout mirrors ``process_local_image()``:
    - A VRT at ``output_file`` listing all *primary* + *ancillary* bands.
    - Individual compressed GeoTIFFs in ``<stem>.data/`` (one per output field).

    Args:
        model (PyTSEB): Fully configured ``PyTSEB`` instance (call
            ``PyTSEB(params)`` before this function).
        output_file (Path): Destination ``.vrt`` path (same as
            ``params['output_file']``).
        block_size (int): Tile side length in pixels. Default 1024 (≈400 MB
            peak for a 20 m S2 tile).
        lst_datetime (datetime | None): UTC datetime of the LST / S3 overpass.
            When provided it is stored as the ``LST_DATETIME`` GDAL metadata
            tag (ISO-8601 format) on every output GeoTIFF and on the VRT.

    Returns:
        Path: ``output_file``
    """
    output_file = Path(output_file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # Fields that receive special treatment and are NOT read via the generic
    # raster-or-scalar path below.
    _SKIP_GENERIC = frozenset(
        ('input_mask', 'SZA', 'SAA', 'G', 'KN_b', 'KN_c', 'KN_C_dash'))

    _open_ds = {}  # field → GDAL dataset (kept open for reuse)

    def _read_block(field_or_path, xoff, yoff, xsize, ysize,
                    dtype=np.float32):
        """Return (ysize, xsize) array from a raster file or broadcast scalar."""
        val = (model.p.get(field_or_path, field_or_path)
               if isinstance(field_or_path, str) and field_or_path in model.p
               else field_or_path)
        # Try scalar first
        try:
            return np.full((ysize, xsize), float(val), dtype=dtype)
        except (ValueError, TypeError):
            pass
        # Raster path
        key = str(val)
        if key not in _open_ds:
            ds = gdal.Open(key, gdal.GA_ReadOnly)
            if ds is None:
                raise FileNotFoundError(f'Cannot open raster: {val}')
            _open_ds[key] = ds
        return (_open_ds[key]
                .GetRasterBand(1)
                .ReadAsArray(xoff, yoff, xsize, ysize)
                .astype(dtype))

    def _try_read_block(field, xoff, yoff, xsize, ysize, dtype=np.float32):
        """Like _read_block but returns None when the field is absent/invalid."""
        try:
            return _read_block(field, xoff, yoff, xsize, ysize, dtype)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Reference raster → image dimensions and geo-metadata
    # ------------------------------------------------------------------
    input_fields = model._get_input_structure()
    ref_path = model.p[next(iter(input_fields))]  # e.g. T_R1
    ref_ds = gdal.Open(str(ref_path), gdal.GA_ReadOnly)
    if ref_ds is None:
        raise FileNotFoundError(f'Cannot open reference input: {ref_path}')
    prj = ref_ds.GetProjection()
    geo = ref_ds.GetGeoTransform()
    nrows = ref_ds.RasterYSize
    ncols = ref_ds.RasterXSize
    ref_ds = None
    model.prj = prj
    model.geo = geo

    # ------------------------------------------------------------------
    # Determine which output fields will be written (S_P and S_A only)
    # ------------------------------------------------------------------
    # calc_daily_ET depends on whether S_dn_24 resolves to a real file/value
    if not model.calc_daily_ET:
        try:
            val = model.p.get('S_dn_24', '')
            float(val)  # scalar – means no real file
        except (ValueError, TypeError):
            if val:  # non-empty string → path
                model.calc_daily_ET = True

    all_out_fields = model._get_output_structure()
    save_fields = [f for f, sv in all_out_fields.items() if sv in (S_P, S_A)]

    # ------------------------------------------------------------------
    # Pre-create output GeoTIFFs (tiled + compressed)
    # ------------------------------------------------------------------
    data_dir = output_file.parent / (output_file.stem + '.data')
    data_dir.mkdir(parents=True, exist_ok=True)

    tif_opts = [
        'COMPRESS=DEFLATE', 'PREDICTOR=2', 'BIGTIFF=IF_SAFER',
        'TILED=YES', f'BLOCKXSIZE={block_size}', f'BLOCKYSIZE={block_size}',
    ]
    drv = gdal.GetDriverByName('GTiff')
    out_ds = {}
    out_paths = {}
    _OUT_NODATA = -9999.0
    for field in save_fields:
        p = str(data_dir / f'{field}.tif')
        out_paths[field] = p
        ds = drv.Create(p, ncols, nrows, 1, gdal.GDT_Float32, tif_opts)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)
        ds.GetRasterBand(1).SetNoDataValue(_OUT_NODATA)
        out_ds[field] = ds

    # ------------------------------------------------------------------
    # G_form scalar: pre-read so it can be broadcast per block
    # ------------------------------------------------------------------
    g_mode = model.G_form[0][0]
    g_scalar = None
    if g_mode in (_tseb_core.G_CONSTANT, _tseb_core.G_RATIO):
        try:
            g_scalar = float(model.G_form[1])
        except (ValueError, TypeError):
            g_scalar = None  # it's a raster; read per block below

    # KN resistance parameters (scalars or rasters)
    kn_fields = ('KN_b', 'KN_c', 'KN_C_dash')

    # ------------------------------------------------------------------
    # Block loop
    # ------------------------------------------------------------------
    n_blocks = ((nrows + block_size - 1) // block_size *
                (ncols + block_size - 1) // block_size)
    block_idx = 0
    for yoff in range(0, nrows, block_size):
        ysize = min(block_size, nrows - yoff)
        for xoff in range(0, ncols, block_size):
            xsize = min(block_size, ncols - xoff)
            dims = (ysize, xsize)
            block_idx += 1
            logger.debug(f'  TSEB block {block_idx}/{n_blocks} '
                         f'({xoff},{yoff}) {xsize}×{ysize}')

            # ---- Read generic raster/scalar fields ----------------------
            in_data = {}
            for field in input_fields:
                if field in _SKIP_GENERIC:
                    continue
                arr = _try_read_block(field, xoff, yoff, xsize, ysize)
                if arr is not None:
                    in_data[field] = arr

            # ---- Landcover (int32) --------------------------------------
            lc_val = model.p.get('landcover', '0')
            try:
                lc = np.full(dims, int(float(lc_val)), dtype=np.int32)
            except (ValueError, TypeError):
                lc = _read_block(lc_val, xoff, yoff, xsize, ysize,
                                 dtype=np.int32)
            in_data['landcover'] = lc

            # ---- Input mask ---------------------------------------------
            if model.p.get('input_mask') == '0':
                mask = np.ones(dims, dtype=np.int32)
                mask[np.logical_or.reduce((
                    lc == _tseb_res.WATER,
                    lc == _tseb_res.URBAN,
                    lc == _tseb_res.SNOW,
                ))] = 0
            else:
                arr = _try_read_block('input_mask', xoff, yoff, xsize, ysize)
                mask = arr.astype(np.int32) if arr is not None else np.ones(
                    dims, dtype=np.int32)

            # Mask pixels where critical inputs are nodata (-9999) or NaN.
            # Without this, pyTSEB computes ET=0 for no-data LST pixels
            # instead of propagating the missing-data flag.
            # h_C and f_g use -9999 as nodata; include them so that
            # canopy-height / green-fraction nodata also masks the pixel.
            _ND_THRESH = -9000.0
            for _crit in ('T_R1', 'LAI', 'f_c', 'T_A1', 'h_C', 'f_g'):
                if _crit in in_data:
                    _arr = in_data[_crit]
                    mask[_arr <= _ND_THRESH] = 0
                    mask[~np.isfinite(_arr)] = 0

            # ---- LAI cap ------------------------------------------------
            # Values up to ~14 m²/m² occur as outliers in the BIOPAR UDP
            # product.  pyTSEB becomes numerically unstable above ~8 m²/m²;
            # clip to a safe physical maximum before the run.
            _LAI_MAX = 8.0
            if 'LAI' in in_data:
                in_data['LAI'] = np.clip(in_data['LAI'], 0.0, _LAI_MAX)

            # ---- SZA / SAA ----------------------------------------------
            lat = in_data.get('lat')
            lon = in_data.get('lon')
            if lat is not None and lon is not None:
                try:
                    in_data['SZA'], in_data['SAA'] = calc_sun_angles(
                        lat, lon,
                        in_data.get('stdlon',
                                    np.zeros(dims, dtype=np.float32)),
                        in_data.get('DOY',
                                    np.zeros(dims, dtype=np.float32)),
                        in_data.get('time',
                                    np.zeros(dims, dtype=np.float32)),
                    )
                except Exception as exc:
                    logger.warning(f'SZA/SAA calculation failed: {exc}')

            # ---- L_dn (longwave irradiance) — estimated if not provided --
            if 'L_dn' not in in_data:
                try:
                    from pyTSEB import net_radiation as _rad
                    in_data['L_dn'] = _rad.calc_longwave_irradiance(
                        in_data['ea'], in_data['T_A1'],
                        in_data['p'], in_data['z_T'])
                except Exception as exc:
                    logger.debug(f'L_dn estimation failed: {exc}')

            # ---- G form -------------------------------------------------
            if g_mode in (_tseb_core.G_CONSTANT, _tseb_core.G_RATIO):
                if g_scalar is not None:
                    g_arr = np.full(dims, g_scalar, dtype=np.float32)
                else:
                    g_arr = _read_block(model.G_form[1],
                                        xoff, yoff, xsize, ysize)
                saved_G_form = model.G_form
                model.G_form = [model.G_form[0], g_arr]
            else:
                saved_G_form = None

            # ---- KN resistance params -----------------------------------
            saved_res_params = model.res_params
            model.res_params = {}
            for kf in kn_fields:
                arr = _try_read_block(kf, xoff, yoff, xsize, ysize)
                if arr is not None:
                    model.res_params[kf] = arr

            # ---- Run the model ------------------------------------------
            out_block = model.run(in_data, mask)

            # ---- Restore mutable model state ----------------------------
            if saved_G_form is not None:
                model.G_form = saved_G_form
            model.res_params = saved_res_params

            # ---- Mask flag=255 (F_INVALID) pixels -----------------------
            # pyTSEB stores the last partially-computed values for pixels
            # where the iteration diverged or T_C exceeded T_C_max.  These
            # "garbage" values (e.g. H_C1 = 60 MW/m²) must be replaced by
            # nodata before writing; only the flag band itself is kept.
            flag_block = out_block.get('flag')
            if flag_block is not None:
                _invalid = flag_block == 255  # pyTSEB F_INVALID constant
                if np.any(_invalid):
                    # --------------------------------------------------
                    # Diagnose F_INVALID pixels.
                    #
                    # pyTSEB sets flag=255 in calc_T_S when:
                    #   T_R^4 - f_theta * T_C^4 < 0
                    # i.e. the iteratively estimated canopy temperature T_C
                    # is physically inconsistent with the observed LST T_R.
                    # Typical causes:
                    #   • LST < T_air (cloud contamination / bad retrieval)
                    #   • Very high LAI → f_theta ≈ 1 → T_S inversion fails
                    #   • T_C diverged (wind too low → R_x → ∞)
                    #
                    # Reported at DEBUG level; raise to WARNING if needed.
                    # --------------------------------------------------
                    n_inv = int(_invalid.sum())
                    n_msk = int(mask.sum())
                    _diag = [
                        f'F_INVALID={n_inv}/{n_msk} valid px '
                        f'in block (xoff={xoff}, yoff={yoff})'
                    ]
                    # ---- key inputs for failing pixels ------------------
                    _inp_labels = {
                        'T_R1': 'LST T_R1 (K)',
                        'T_A1': 'T_air T_A1 (K)',
                        'LAI':  'LAI (m²/m²)',
                        'f_c':  'f_cover',
                        'f_g':  'f_green',
                        'SZA':  'SZA (°)',
                        'u':    'wind (m/s)',
                        'S_dn': 'S_dn (W/m²)',
                    }
                    for _fld, _lbl in _inp_labels.items():
                        _arr = in_data.get(_fld)
                        if _arr is not None and _arr.shape == _invalid.shape:
                            _v = _arr[_invalid]
                            _v = _v[np.isfinite(_v)]
                            if _v.size:
                                _diag.append(
                                    f'  {_lbl:22s}: '
                                    f'mean={_v.mean():8.2f}  '
                                    f'min={_v.min():8.2f}  '
                                    f'max={_v.max():8.2f}')
                    # ---- T_C from pyTSEB output (the direct trigger) ----
                    # T_temp = T_R^4 - f_theta*T_C^4 < 0 → F_INVALID
                    for _ofld, _olbl in [('T_C', 'T_C iterative (K)'),
                                         ('T_S', 'T_S iterative (K)')]:
                        _oa = out_block.get(_ofld)
                        if _oa is not None and _oa.shape == _invalid.shape:
                            _v = _oa[_invalid]
                            _v = _v[np.isfinite(_v)]
                            if _v.size:
                                _diag.append(
                                    f'  {_olbl:22s}: '
                                    f'mean={_v.mean():8.2f}  '
                                    f'min={_v.min():8.2f}  '
                                    f'max={_v.max():8.2f}')
                    logger.debug('\n'.join(_diag))
                    for _field in save_fields:
                        if _field != 'flag' and _field in out_block:
                            out_block[_field] = np.where(
                                _invalid, np.nan, out_block[_field])

            # ---- Write output blocks ------------------------------------
            for field in save_fields:
                if field in out_block:
                    data = out_block[field].astype(np.float32)
                    # Replace NaN/inf with the nodata sentinel so that
                    # pixels excluded by the input mask (or that failed to
                    # converge inside pyTSEB) are stored as proper nodata
                    # rather than 0 or ±inf.
                    data[~np.isfinite(data)] = _OUT_NODATA
                    out_ds[field].GetRasterBand(1).WriteArray(
                        data, xoff, yoff)

    # ------------------------------------------------------------------
    # Flush and close output datasets
    # ------------------------------------------------------------------
    if lst_datetime is not None:
        lst_dt_str = lst_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        for ds in out_ds.values():
            ds.SetMetadataItem('LST_DATETIME', lst_dt_str)
    for ds in out_ds.values():
        ds.FlushCache()
    out_ds.clear()

    # Close input datasets
    for ds in _open_ds.values():
        del ds
    _open_ds.clear()

    # ------------------------------------------------------------------
    # Build VRT listing all saved bands (one per output field)
    # gdal.BuildVRT with separate=True stacks each TIF as its own band.
    # We then annotate band descriptions via the VRT XML API.
    # ------------------------------------------------------------------
    vrt_ds = gdal.BuildVRT(
        str(output_file),
        [out_paths[f] for f in save_fields],
        separate=True,
    )
    if vrt_ds is None:
        raise RuntimeError(f'gdal.BuildVRT failed for {output_file}')
    for i, field in enumerate(save_fields, start=1):
        vrt_ds.GetRasterBand(i).SetDescription(field)
    if lst_datetime is not None:
        vrt_ds.SetMetadataItem('LST_DATETIME',
                               lst_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'))
    vrt_ds.FlushCache()
    vrt_ds = None
    logger.info(f'TSEB VRT written: {output_file}')
    return output_file


def plot_et_histogram(vrt_file: Path, et_field: str = 'ET_day',
                      outfile: Path | None = None) -> Path:
    """Generate and save a histogram of ET values from a TSEB-PT output VRT.

    The histogram gives a quick sanity check for nonsense ET calculations.
    Valid pixels only (nodata excluded); statistics (mean, median, p5/p95)
    are overlaid.

    Args:
        vrt_file (Path): TSEB-PT output VRT (as written by
            :func:`_process_tseb_tiled`).  The individual per-field GeoTIFFs
            are expected alongside it in ``<vrt_stem>.data/``.
        et_field (str): Name of the ET field to plot.  Defaults to
            ``'ET_day'`` (mm/day).  Falls back to ``'LE'`` (W/m²) if the
            field is not found.
        outfile (Path | None): Where to save the PNG.  Defaults to
            ``<vrt_file>.png`` (same directory, same stem).

    Returns:
        Path: Path to the saved PNG file.
    """
    vrt_file = Path(vrt_file)
    data_dir = vrt_file.parent / (vrt_file.stem + '.data')

    # Locate the requested field GeoTIFF
    tif_path = data_dir / f'{et_field}.tif'
    if not tif_path.exists():
        fallback = 'LE'
        logger.warning(
            f'ET field "{et_field}" not found in {data_dir}, '
            f'falling back to "{fallback}"'
        )
        et_field = fallback
        tif_path = data_dir / f'{et_field}.tif'
    if not tif_path.exists():
        raise FileNotFoundError(
            f'Cannot find ET GeoTIFF for field "{et_field}" in {data_dir}')

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata

    # Mask nodata and NaN
    valid = data != nodata if nodata is not None else np.ones(data.shape, bool)
    valid &= np.isfinite(data)
    values = data[valid]

    unit = 'mm/day' if et_field == 'ET_day' else 'W/m²'
    n_valid = values.size
    n_total = data.size
    pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))

    if n_valid == 0:
        ax.text(0.5, 0.5, 'No valid pixels', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='red')
    else:
        p5, p25, med, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        mean = float(np.mean(values))
        std  = float(np.std(values))

        # Clip display range to p5–p95 to avoid extreme outliers dominating
        ax.hist(values, bins=100, range=(p5, p95),
                color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(mean, color='tomato',  lw=1.5, ls='-',  label=f'Mean  {mean:.2f}')
        ax.axvline(med,  color='orange',  lw=1.5, ls='--', label=f'Median {med:.2f}')
        ax.axvline(p5,   color='grey',    lw=1.0, ls=':',  label=f'P5    {p5:.2f}')
        ax.axvline(p95,  color='grey',    lw=1.0, ls=':',  label=f'P95   {p95:.2f}')
        ax.legend(fontsize=8, loc='upper right')

        stats_txt = (
            f'n valid: {n_valid:,} / {n_total:,} ({pct_valid:.1f}%)\n'
            f'mean ± std: {mean:.2f} ± {std:.2f} {unit}\n'
            f'min / max: {values.min():.2f} / {values.max():.2f} {unit}'
        )
        ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
                va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    title = vrt_file.stem
    ax.set_title(f'{title}\n{et_field} histogram (display range: p5–p95)',
                 fontsize=9)
    ax.set_xlabel(f'{et_field} ({unit})')
    ax.set_ylabel('Pixel count')
    fig.tight_layout()

    if outfile is None:
        outfile = vrt_file.with_suffix('.png')
    outfile = Path(outfile)
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    logger.info(f'ET histogram saved: {outfile}')
    return outfile


def compute_et(tile, time, lst_file, vza_file, lat_file, lon_file, elev_file,
               biopar_dict, worldcover_file, meteo_paths, outdir, time_zone,
               biopar_cache_dir=None, et_histogram: bool = False):
    """
    Run the TSEB-PT model via pyTSEB to compute evapotranspiration.

    F_G (fraction green) and H_C (canopy height) are derived from the biopar
    and WorldCover land-cover using :func:`calc_fg` and
    :func:`calc_canopy_height`.  Both outputs are cached in
    ``biopar_cache_dir`` (defaults to ``outdir/biopar/``) so they are only
    computed once and can be shared between the standard-ET and LSTM-ET runs.

    Args:
        tile (str): MGRS tile identifier (used in the output filename).
        time (datetime): S3 overpass datetime.
        lst_file (Path): Sharpened LST GeoTIFF (T_R1 input).  May be any
            integer-scaled or float GeoTIFF — the rasterio scale factor is
            applied automatically to convert to physical Kelvin before the
            file is passed to pyTSEB.
        vza_file (Path): View zenith angle GeoTIFF at 20 m (degrees).
        lat_file (Path): Latitude grid GeoTIFF at 20 m (degrees).
        lon_file (Path): Longitude grid GeoTIFF at 20 m (degrees).
        elev_file (Path): DEM GeoTIFF at 20 m (m).
        biopar_dict (dict): ``{variable: {datetime: Path}}`` for at least
            ``'LAI'``, ``'FAPAR'`` and ``'FCOVER'``.
        worldcover_file (Path): ESA WorldCover GeoTIFF warped to S2 20 m grid.
        meteo_paths (dict): Output of :func:`compute_meteo_for_tseb`.
        outdir (Path): Directory to write the TSEB output VRT to.
        time_zone (int): UTC offset in whole hours.
        biopar_cache_dir (Path | None): Directory used to cache F_G, H_C and
            the remapped landcover raster.  Defaults to ``outdir / 'biopar'``.
            Pass the standard-ET biopar dir here when calling for LSTM-ET so
            that the expensive rasters are not recomputed.
        et_histogram (bool): If ``True``, call :func:`plot_et_histogram` after
            the model run and save a PNG histogram of ``ET_day`` next to the
            VRT (``<VRT_stem>.png``).  Defaults to ``False``.

    Returns:
        Path: Path to the output VRT file containing all TSEB output bands
        (including ``ET_day`` in mm/day).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    timestr  = time.strftime('%Y%m%dT%H%M%S')
    doy      = time.timetuple().tm_yday
    time_utc = time.hour + time.minute / 60.0 + time.second / 3600.0

    # Match closest biopar date to the S3 overpass time
    lai_dates = sorted(biopar_dict['LAI'].keys())
    closest   = min(lai_dates,
                    key=lambda d: abs((d - time).total_seconds()))

    # F_G, H_C and landcover remap are per biopar date — compute once and cache
    if biopar_cache_dir is None:
        biopar_cache_dir = outdir / 'biopar'
    biopar_cache_dir.mkdir(parents=True, exist_ok=True)
    closest_str   = closest.strftime('%Y%m%d')
    doy_biopar    = closest.timetuple().tm_yday

    fg_file = biopar_cache_dir / f'{closest_str}_FG.tif'
    hc_file = biopar_cache_dir / f'{closest_str}_HC.tif'

    if not fg_file.exists():
        logger.info(f'Computing F_G for biopar date {closest_str}')
        calc_fg(biopar_dict['FAPAR'][closest], biopar_dict['LAI'][closest],
                doy_biopar, fg_file)

    if not hc_file.exists():
        logger.info(f'Computing H_C for biopar date {closest_str}')
        calc_canopy_height(biopar_dict['LAI'][closest], worldcover_file,
                           fg_file, hc_file)

    # Remap WorldCover to pyTSEB internal landcover codes (cached once)
    lc_file = biopar_cache_dir / 'landcover_pytseb.tif'
    remap_worldcover_for_pytseb(worldcover_file, lc_file)

    # PyTSEB reads rasters with GDAL ReadAsArray(), which does NOT apply the
    # rasterio scale tag.  LST is stored as uint16 (scale=0.01) so we must
    # write a float32 version in physical Kelvin before passing it.
    lst_k_file = outdir / f'{timestr}_T_R1_K.tif'
    if not lst_k_file.exists():
        with rasterio.open(lst_file) as src:
            raw  = src.read(1).astype(np.float32)
            scl  = src.scales[0] if src.scales else 1.0
            nd   = src.nodata
            prof = src.profile.copy()
        lst_k = raw * scl
        if nd is not None:
            lst_k[raw == nd] = -9999.0
        prof.update(dtype=rasterio.float32, nodata=-9999.0, compress='deflate', PREDICTOR=2)
        with rasterio.open(lst_k_file, 'w', **prof) as dst:
            dst.write(lst_k, 1)

    output_file = outdir / f'TSEB-PT_{timestr}_{tile}.vrt'

    params = {
        'model':       'TSEB_PT',
        'output_file': str(output_file),
        # --- Radiometric input ---
        'T_R1':       str(lst_k_file),
        'VZA':        str(vza_file),
        'landcover':  str(lc_file),
        'input_mask': '0',
        # --- Vegetation structure ---
        'LAI':    str(biopar_dict['LAI'][closest]),
        'f_c':    str(biopar_dict['FCOVER'][closest]),
        'f_g':    str(fg_file),   # fraction green: FAPAR/LAI/SZA iterative
        'h_C':    str(hc_file),   # canopy height: LAI + WorldCover LUT
        'w_C':    1.0,    # leaf width / canopy height ratio
        # --- Geometry ---
        'lat':    str(lat_file),
        'lon':    str(lon_file),
        'alt':    str(elev_file),
        'stdlon': time_zone * 15.0,
        # --- Timing ---
        'DOY':  doy,
        'time': time_utc,
        # --- Meteorology at 100 m blending height ---
        'z_T':   100.0,
        'z_u':   100.0,
        'T_A1':  str(meteo_paths['T_A1']),
        'u':     str(meteo_paths['u']),
        'p':     str(meteo_paths['p']),
        'ea':    str(meteo_paths['ea']),
        'S_dn':    str(meteo_paths['S_dn']),
        'S_dn_24': str(meteo_paths['S_dn_24']),
        # --- Leaf optical properties (broadleaf defaults) ---
        'emis_C':    0.99,
        'emis_S':    0.97,
        'rho_vis_C': 0.07,
        'tau_vis_C': 0.08,
        'rho_nir_C': 0.32,
        'tau_nir_C': 0.33,
        'rho_vis_S': 0.15,
        'rho_nir_S': 0.25,
        # --- Canopy structure ---
        'alpha_PT':   1.26,
        'x_LAD':      1.0,   # spherical leaf angle distribution
        'z0_soil':    0.01,
        'leaf_width': 0.05,
        # --- Turbulence / resistance ---
        'resistance_form': 0,
        'KN_b':      0.012,
        'KN_c':      0.0038,
        'KN_C_dash': 90,
        # --- Soil heat flux: constant ratio G = 0.35 * Rn ---
        'G_form': [[1], 0.35],
        # --- Misc ---
        'water_stress': 0,
        'calc_row': [0, 0],
        'row_az':   0,
    }

    logger.info(f'Running TSEB-PT for {timestr}')
    model = PyTSEB(params)
    _process_tseb_tiled(model, output_file, lst_datetime=time)
    del model
    gc.collect()

    # Clean up temporary intermediate files to save storage
    if lst_k_file.exists():
        logger.debug(f'Removing temporary LST conversion file: {lst_k_file.name}')
        lst_k_file.unlink()
    
    # Clean up meteo intermediate GeoTIFFs (only needed for TSEB input)
    meteo_cleanup = [
        outdir / f'{timestr}_TA.tif',
        outdir / f'{timestr}_WS.tif',
        outdir / f'{timestr}_EA.tif',
        outdir / f'{timestr}_PA.tif',
        outdir / f'{timestr}_SW-IN.tif',
        outdir / f'{datestr}_SW-IN-DD.tif',
    ]
    for meteo_file in meteo_cleanup:
        if meteo_file.exists():
            logger.debug(f'Removing temporary meteo file: {meteo_file.name}')
            meteo_file.unlink()

    if et_histogram:
        plot_et_histogram(output_file)

    return output_file
