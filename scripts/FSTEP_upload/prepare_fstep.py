"""Prepare FSTEP CSV metadata files for uploading GeoTIFFs to Food Security Explorer.

This module generates the CSV metadata files required by the FSTEP upload process.
It reads produced GeoTIFF files and extracts timestamps and product information.

Usage:
    from prepare_fstep import generate_lst_ta_csv, generate_ndvi_csv, generate_et_csv
    
    generate_lst_ta_csv(
        geotiff_dir=Path("./results/35LPD/005_lst-ta"),
        output_csv=Path("./results/35LPD/FSTEP_upload_lst-ta_35LPD.csv")
    )
    generate_ndvi_csv(
        geotiff_dir=Path("./results/35LPD/006_ndvi"),
        output_csv=Path("./results/35LPD/FSTEP_upload_NDVI_35LPD.csv")
    )
    generate_et_csv(
        geotiff_dir=Path("./results/35LPD/007_et"),
        output_csv=Path("./results/35LPD/FSTEP_upload_ET_35LPD.csv")
    )
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import re


def _extract_timestamp_from_filename(filename: str, pattern: str) -> Optional[pd.Timestamp]:
    """Extract ISO timestamp from filename using the given regex pattern.
    
    Parameters
    ----------
    filename : str
        The filename to parse.
    pattern : str
        Regex pattern with a capture group for the timestamp.
        E.g., r"(\d{8}T\d{6})" to match YYYYMMDDThhmmss
    
    Returns
    -------
    Optional[pd.Timestamp]
        Parsed timestamp or None if pattern doesn't match.
    """
    m = re.search(pattern, filename)
    if not m:
        return None
    try:
        ts_str = m.group(1)
        # Convert YYYYMMDDThhmmss -> YYYY-MM-DD
        if len(ts_str) == 15 and 'T' in ts_str:  # YYYYMMDDThhmmss format
            return pd.to_datetime(ts_str, format="%Y%m%dT%H%M%S")
        else:
            return pd.to_datetime(ts_str)
    except Exception:
        return None


def generate_lst_ta_csv(
    geotiff_dir: Path,
    output_csv: Path,
    tile: Optional[str] = None,
) -> None:
    """Generate FSTEP CSV metadata for LST-Ta GeoTIFFs.
    
    Parameters
    ----------
    geotiff_dir : Path
        Directory containing LST-Ta GeoTIFF files.
    output_csv : Path
        Path to write the CSV file.
    tile : Optional[str]
        Tile name (extracted from filename if not provided).
    """
    geotiff_dir = Path(geotiff_dir)
    output_csv = Path(output_csv)
    
    files = sorted(geotiff_dir.glob("LST-Ta_*.tif"))
    if not files:
        raise ValueError(f"No LST-Ta GeoTIFF files found in {geotiff_dir}")
    
    rows = []
    for f in files:
        ts = _extract_timestamp_from_filename(f.name, r"LST-Ta_(\d{8}T\d{6})_")
        if ts is None:
            continue
        
        if tile is None and "_" in f.name:
            parts = f.name.split("_")
            if len(parts) >= 3:
                tile = parts[2].replace(".tif", "")
        
        rows.append({
            "geometry": "",
            "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filename": f.name,
            "description": "Land Surface Temperature - air temperature (K)",
        })
    
    if not rows:
        raise ValueError(f"Could not extract timestamps from LST-Ta files in {geotiff_dir}")
    
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"LST-Ta CSV written to {output_csv} ({len(rows)} files)")


def generate_ndvi_csv(
    geotiff_dir: Path,
    output_csv: Path,
    tile: Optional[str] = None,
) -> None:
    """Generate FSTEP CSV metadata for NDVI GeoTIFFs.
    
    Parameters
    ----------
    geotiff_dir : Path
        Directory containing NDVI GeoTIFF files.
    output_csv : Path
        Path to write the CSV file.
    tile : Optional[str]
        Tile name (extracted from filename if not provided).
    """
    geotiff_dir = Path(geotiff_dir)
    output_csv = Path(output_csv)
    
    files = sorted(geotiff_dir.glob("NDVI_*.tif"))
    if not files:
        raise ValueError(f"No NDVI GeoTIFF files found in {geotiff_dir}")
    
    rows = []
    for f in files:
        # NDVI files use YYYY-MM-DD format
        ts = _extract_timestamp_from_filename(f.name, r"NDVI_(\d{4}-\d{2}-\d{2})")
        if ts is None:
            continue
        
        if tile is None and "_" in f.name:
            parts = f.name.split("_")
            if len(parts) >= 3:
                tile = parts[2].replace(".tif", "")
        
        end_time = ts + pd.Timedelta("1D")
        rows.append({
            "geometry": "",
            "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filename": f.name,
            "description": "Normalized difference vegetation index",
        })
    
    if not rows:
        raise ValueError(f"Could not extract timestamps from NDVI files in {geotiff_dir}")
    
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"NDVI CSV written to {output_csv} ({len(rows)} files)")


def generate_et_csv(
    geotiff_dir: Path,
    output_csv: Path,
    tile: Optional[str] = None,
    product_name: str = "TSEB-PT",
) -> None:
    """Generate FSTEP CSV metadata for ET GeoTIFFs (VRT or GeoTIFF).
    
    Parameters
    ----------
    geotiff_dir : Path
        Directory containing ET GeoTIFF/VRT files.
    output_csv : Path
        Path to write the CSV file.
    tile : Optional[str]
        Tile name (extracted from filename if not provided).
    product_name : str
        Product name for the description (default: TSEB-PT).
    """
    geotiff_dir = Path(geotiff_dir)
    output_csv = Path(output_csv)
    
    # Look for both .tif and .vrt files
    files = sorted(list(geotiff_dir.glob(f"{product_name}_*.tif")) + 
                   list(geotiff_dir.glob(f"{product_name}_*.vrt")))
    if not files:
        raise ValueError(f"No ET GeoTIFF/VRT files found in {geotiff_dir}")
    
    rows = []
    for f in files:
        ts = _extract_timestamp_from_filename(f.name, rf"{product_name}_(\d{{8}}T\d{{6}})_")
        if ts is None:
            continue
        
        if tile is None and "_" in f.name:
            parts = f.name.split("_")
            if len(parts) >= 3:
                tile = parts[2].replace(".tif", "").replace(".vrt", "")
        
        rows.append({
            "geometry": "",
            "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filename": f.name,
            "description": f"{product_name} Evapotranspiration (mm/day)",
        })
    
    if not rows:
        raise ValueError(f"Could not extract timestamps from ET files in {geotiff_dir}")
    
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"ET CSV written to {output_csv} ({len(rows)} files)")


def prepare_all_fstep_csvs(
    output_dir: Path,
    tile: str,
    skip_missing: bool = True,
) -> dict:
    """Prepare all FSTEP CSV files for a tile.
    
    Parameters
    ----------
    output_dir : Path
        Root output directory containing per-tile folders.
    tile : str
        Sentinel-2 tile ID (e.g., "35LPD").
    skip_missing : bool
        If True, skip products with missing directories. If False, raise error.
    
    Returns
    -------
    dict
        Map of product name to CSV path for successfully generated files.
    """
    results = {}
    
    # LST-Ta
    lst_ta_dir = output_dir / tile / "005_lst-ta"
    if lst_ta_dir.exists():
        try:
            csv_path = output_dir / tile / f"FSTEP_upload_lst-ta_{tile}.csv"
            generate_lst_ta_csv(lst_ta_dir, csv_path, tile=tile)
            results["LST-Ta"] = csv_path
        except Exception as e:
            if not skip_missing:
                raise
            print(f"Warning: Could not generate LST-Ta CSV: {e}")
    elif not skip_missing:
        raise ValueError(f"LST-Ta directory not found: {lst_ta_dir}")
    
    # NDVI
    ndvi_dir = output_dir / tile / "006_ndvi"
    if ndvi_dir.exists():
        try:
            csv_path = output_dir / tile / f"FSTEP_upload_NDVI_{tile}.csv"
            generate_ndvi_csv(ndvi_dir, csv_path, tile=tile)
            results["NDVI"] = csv_path
        except Exception as e:
            if not skip_missing:
                raise
            print(f"Warning: Could not generate NDVI CSV: {e}")
    elif not skip_missing:
        raise ValueError(f"NDVI directory not found: {ndvi_dir}")
    
    # ET
    et_dir = output_dir / tile / "007_et"
    if et_dir.exists():
        try:
            csv_path = output_dir / tile / f"FSTEP_upload_ET_{tile}.csv"
            generate_et_csv(et_dir, csv_path, tile=tile, product_name="TSEB-PT")
            results["ET"] = csv_path
        except Exception as e:
            if not skip_missing:
                raise
            print(f"Warning: Could not generate ET CSV: {e}")
    elif not skip_missing:
        raise ValueError(f"ET directory not found: {et_dir}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_fstep.py <output_dir> [tile1] [tile2] ...")
        print("  output_dir: root output directory containing per-tile folders")
        print("  tiles: tile IDs to process (defaults to all subdirectories)")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    tiles = sys.argv[2:] if len(sys.argv) > 2 else None
    
    if not output_dir.exists():
        print(f"Error: output_dir does not exist: {output_dir}")
        sys.exit(1)
    
    if tiles is None:
        # Auto-detect tiles from subdirectories
        tiles = [d.name for d in output_dir.iterdir() if d.is_dir()]
    
    for tile in tiles:
        print(f"\n** Preparing FSTEP CSVs for tile {tile}")
        try:
            results = prepare_all_fstep_csvs(output_dir, tile, skip_missing=True)
            for product, csv_path in results.items():
                print(f"   {product}: {csv_path}")
        except Exception as e:
            print(f"Error preparing CSVs for {tile}: {e}")
