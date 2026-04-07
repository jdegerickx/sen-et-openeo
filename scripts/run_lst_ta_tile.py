
"""
    This script generates timeseries of LST-Ta (Land Surface Temperature -
    air temperature) and NDVI (Normalized Difference Vegetation Index)
    for one or multiple Sentinel-2 tiles.
    The required inputs are downloaded from the Copernicus Data
    Space Ecosystem (CDSE) using OpenEO and from the Copernicus Climate
    Data Store (CDS).

    Before computing the LST-Ta indicator, the LST data from Sentinel-3
    is sharpened to 20m using Sentinel-2 and Copernicus DEM. After that,
    additional bias and directionality correction is done based on
    comparison with ECOSTRESS data. The computation of these correction
    parameters is done in a separate series of Python scripts:
    https://github.com/SnydersLouis/wasdi_ecostress
    """

from pathlib import Path
from pickle import TRUE
import pandas as pd
from loguru import logger
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from scipy.constants import Stefan_Boltzmann as sigma
import shutil

from sen_et_openeo.data_download import SenETDownload
from sen_et_openeo.era5 import (ERA5Collection,
                                ERA5TimeSeriesProcessor,
                                get_default_rsi_meta)
from sen_et_openeo.tseb import compute_meteo_for_tseb, compute_et


def get_era5_data(era5_tiled_folder, tile, temporal_extent):

    timestamps = pd.date_range(pd.to_datetime(temporal_extent[0]),
                               pd.to_datetime(temporal_extent[1]),
                               freq='1D')

    # create ERA5 collection
    era5col = ERA5Collection.from_path(era5_tiled_folder)

    # now check completeness of era5 collection
    # based on tile and dates
    # if no meteo products are available yet,
    # they are downloaded here
    era5col = era5col.check_tiledates(tile, timestamps,
                                      era5_tiled_folder)

    df = era5col.df.copy()
    df['day'] = [pd.to_datetime(d).strftime('%Y-%m-%d')
                 for d in df['date'].values]
    era5col = era5col._clone(df=df)

    era5col = era5col.filter_tiles(tile)

    return era5col


def compute_lst_ta(lst_file, time, elev_file, time_zone,
                   era5col, outfile):

    out_scale = 0.01

    # Get the LST data
    with rasterio.open(lst_file, 'r') as src:
        output_profile = src.profile.copy()
        lst_raw = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        # Use file metadata scale so this works for:
        # - scaled integer LST (e.g. scale=0.01)
        # - float Kelvin LST (scale=1.0 / no scale tag)
        src_scale = src.scales[0] if src.scales and src.scales[0] else 1.0

    lst_data = lst_raw * src_scale
    if src_nodata is not None:
        lst_data[lst_raw == src_nodata] = np.nan

    # Get the air temperature data from ERA5.
    # Use lst_file as the spatial reference template so that ERA5 T_A1 is
    # resampled to the same grid as the LST (works for both the 20 m standard
    # product and the 30 m LSTM-like product).
    meteo_settings = {'bands': ['t2m']}
    meteo_rsi_meta = get_default_rsi_meta().get('ERA5')
    elev = None

    meteo_ts = ERA5TimeSeriesProcessor([time],
                                       elev,
                                       lst_file,
                                       time_zone,
                                       era5col,
                                       meteo_settings,
                                       rsi_meta=meteo_rsi_meta,
                                       ).compute_ts()

    lst_ta = lst_data - np.squeeze(meteo_ts.data)

    newnodata = -999
    lst_ta = lst_ta / out_scale
    lst_ta[np.isnan(lst_ta)] = newnodata
    lst_ta = lst_ta.astype(np.int16)

    output_profile.update(dtype=rasterio.int16, nodata=newnodata)

    # write result to file
    with rasterio.open(outfile, 'w', **output_profile) as dst:
        dst.write(lst_ta, 1)
        dst.scales = [out_scale]
        dst.descriptions = ['LST-Ta']
        dst.units = ['K']
        # Generate overviews
        dst.build_overviews(
            (4, 8, 16), Resampling.average)


def generate_lstm_like_lst(
        sharpened_lst_file: Path,
        vza_file: Path,
        outfile: Path,
        target_resolution: float = 30.0,
        max_vza_deg: float = 30.0):
    """Resample sharpened S3 LST from 20 m to an LSTM-like product at
    ``target_resolution`` metres, masking pixels with VZA > ``max_vza_deg``.

    Conversion path:
        scaled LST → LST (K) → radiance (Stefan-Boltzmann)
        → resample to 30 m (cubic spline)
        → resample VZA to 30 m (nearest)
        → mask VZA > max_vza_deg
        → LST (K) [float32 GeoTIFF, nodata = -9999]

    Parameters
    ----------
    sharpened_lst_file : Path
        Sharpened LST at 20 m.  Scale factor is read from file metadata.
    vza_file : Path
        View Zenith Angle at 20 m (scaled int16, degrees after applying scale).
    outfile : Path
        Output GeoTIFF (float32, LST in Kelvin).
    target_resolution : float
        Target pixel size in metres. Default 30 m.
    max_vza_deg : float
        Maximum allowed VZA in degrees. Pixels above this are masked.
        Default 30°.
    """
    _NODATA = -9999.0

    # --- 1. Read LST and convert to Kelvin ---------------------------------
    with rasterio.open(sharpened_lst_file) as src:
        lst_raw = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        lst_scale = src.scales[0] if src.scales else 0.01
        src_transform = src.transform
        src_crs = src.crs
        src_height, src_width = src.height, src.width

    lst_k = lst_raw * lst_scale
    if src_nodata is not None:
        lst_k[lst_raw == src_nodata] = np.nan

    # --- 2. Convert to radiance (Stefan-Boltzmann: R = sigma * T^4) --------
    radiance = np.full((src_height, src_width), _NODATA, dtype=np.float32)
    valid = ~np.isnan(lst_k) & (lst_k > 0)
    radiance[valid] = sigma * (lst_k[valid] ** 4)

    # --- 3. Compute target grid at requested resolution --------------------
    left = src_transform.c
    top = src_transform.f
    right = left + src_transform.a * src_width
    bottom = top + src_transform.e * src_height  # transform.e is negative
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, src_crs, src_width, src_height,
        left=left, bottom=bottom, right=right, top=top,
        resolution=target_resolution,
    )

    # --- 4. Resample radiance to target resolution (cubic spline) ----------
    radiance_30m = np.full((dst_height, dst_width), _NODATA, dtype=np.float32)
    reproject(
        source=radiance,
        destination=radiance_30m,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=_NODATA,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_nodata=_NODATA,
        resampling=Resampling.cubic_spline,
    )

    # --- 5. Resample VZA to target resolution (nearest) -------------------
    with rasterio.open(vza_file) as vsrc:
        vza_raw = vsrc.read(1).astype(np.float32)
        vza_scale = vsrc.scales[0] if vsrc.scales else 0.01
        vza_nodata = vsrc.nodata
        vza_transform = vsrc.transform
        vza_crs = vsrc.crs

    vza_deg = vza_raw * vza_scale
    _VZA_NODATA = -9999.0
    if vza_nodata is not None:
        vza_deg[vza_raw == vza_nodata] = _VZA_NODATA

    vza_30m = np.full((dst_height, dst_width), _VZA_NODATA, dtype=np.float32)
    reproject(
        source=vza_deg,
        destination=vza_30m,
        src_transform=vza_transform,
        src_crs=vza_crs,
        src_nodata=_VZA_NODATA,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_nodata=_VZA_NODATA,
        resampling=Resampling.nearest,
    )

    # --- 6. Apply VZA mask ------------------------------------------------
    vza_bad = (vza_30m == _VZA_NODATA) | (vza_30m > max_vza_deg)
    radiance_30m[vza_bad] = _NODATA

    # --- 7. Convert radiance back to LST (K): T = (R / sigma)^(1/4) ------
    lst_30m = np.full((dst_height, dst_width), _NODATA, dtype=np.float32)
    valid_out = radiance_30m != _NODATA
    lst_30m[valid_out] = (radiance_30m[valid_out] / sigma) ** 0.25

    # --- 8. Write output --------------------------------------------------
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': dst_width,
        'height': dst_height,
        'count': 1,
        'crs': src_crs,
        'transform': dst_transform,
        'nodata': _NODATA,
        'compress': 'deflate',
    }
    with rasterio.open(outfile, 'w', **out_profile) as dst:
        dst.write(lst_30m, 1)
        dst.update_tags(1, description='LST LSTM-like', units='K')


def resample_to_match(src_file: Path, ref_file: Path, dst_file: Path,
                      resampling: Resampling = Resampling.bilinear) -> Path:
    """Reproject/resample *src_file* to exactly match the grid of *ref_file*.

    Uses the CRS, transform, width and height of *ref_file* as the target.
    Returns *dst_file* immediately if it already exists (lazy caching).
    Preserves rasterio scale and description metadata from *src_file*.
    """
    dst_file = Path(dst_file)
    if dst_file.exists():
        return dst_file
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_file) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
    with rasterio.open(src_file) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        src_nd_out = src_nodata if src_nodata is not None else -9999.0
        src_tr = src.transform
        src_crs = src.crs
        src_scales = src.scales
        src_descriptions = src.descriptions
    dst_data = np.full((dst_height, dst_width), src_nd_out, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_tr,
        src_crs=src_crs,
        src_nodata=src_nd_out,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=src_nd_out,
        resampling=resampling,
    )
    dst_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': dst_width,
        'height': dst_height,
        'count': 1,
        'crs': dst_crs,
        'transform': dst_transform,
        'nodata': src_nd_out,
        'compress': 'deflate',
    }
    with rasterio.open(dst_file, 'w', **dst_profile) as dst:
        dst.write(dst_data, 1)
        if src_scales:
            dst.scales = src_scales
        if src_descriptions:
            dst.descriptions = src_descriptions
    return dst_file


def _find_external_lst(folder: Path, tile: str, timestamp) -> Path:
    """Locate an external LST GeoTIFF by matching on date (YYYYMMDD) only.

    Searches:
        <folder>/<tile>/<year>/S3-LSTHR>/*/
            <tile>_S3-LSTHR_<YYYYMMDD>*_COG.tif

    If multiple overpasses exist on the same day, the file whose parent
    directory name (the overpass timestamp) is closest to *timestamp* is
    returned.  Returns None if no match is found.
    """
    datestr = timestamp.strftime('%Y%m%d')
    year = timestamp.strftime('%Y')
    search_dir = Path(folder) / tile / year / 'S3-LSTHR'
    if not search_dir.exists():
        return None
    matches = sorted(search_dir.glob(
        f'*/{tile}_S3-LSTHR_{datestr}*_COG.tif'))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Multiple overpasses on the same day — pick closest to the S3 timestamp
    def _parse_dir_ts(p):
        try:
            return pd.Timestamp(p.parent.name)
        except Exception:
            return pd.NaT
    return min(matches,
               key=lambda p: abs((_parse_dir_ts(p) - timestamp)
                                 .total_seconds()))


def main(tile, temporal_extent, time_zone, output_dir, era5_tiled_folder,
         residual_correction=False, corr_parameters=None,
         parallel_jobs=False, delete_tmp_data=False,
         generate_lstm_like=False, et_histogram=False,
         compute_et_tseb=True,
         min_valid_s3_fraction=0.0,
         mask_to_s3_coverage=False,
         external_lst_folder=None,
         biopar_chunk_months=1,
         s2_chunk_months=1,
         max_concurrent_jobs=4):

    logger.info('** Downloading data from OpenEO')
    data_download = SenETDownload(tile, temporal_extent)
    data_download.download(output_dir,
                           parallel=parallel_jobs,
                           download_biopar=compute_et_tseb,
                           biopar_chunk_months=biopar_chunk_months,
                           s2_chunk_months=s2_chunk_months,
                           max_concurrent_jobs=max_concurrent_jobs)

    logger.info('** Preprocessing data')
    preprocess_dict = data_download.preprocess(
        output_dir, delete_unrequired_data=delete_tmp_data)

    logger.info('** Running LST sharpening algorithm')
    sharpening_dict = data_download.sharpening(
        output_dir, residual_correction,
        min_valid_fraction=min_valid_s3_fraction,
        mask_to_s3_coverage=mask_to_s3_coverage)

    if corr_parameters is not None:
        logger.info('** Apply LST correction')
        lst_dict = data_download.lst_correction(
            output_dir, corr_parameters)
    else:
        lst_dict = sharpening_dict

    logger.info('** Getting ERA5 data')
    era5col = get_era5_data(era5_tiled_folder, tile, temporal_extent)

    logger.info('** Computing LST-Ta')
    elev_file = preprocess_dict['COPERNICUS_30']['alt']
    timestamps = list(lst_dict['LST'].keys())
    outdir = output_dir / tile / '005_lst-ta'
    outdir.mkdir(parents=True, exist_ok=True)
    outfiles = []
    for t in timestamps:
        lst_file = lst_dict['LST'][t]
        time = t.strftime('%Y%m%dT%H%M%S')
        outfile = outdir / f'LST-Ta_{time}_{tile}.tif'
        outfiles.append(outfile)
        if Path(outfile).exists():
            continue
        compute_lst_ta(lst_file, t, elev_file, time_zone,
                       era5col, outfile)

    # create pandas dataframe with needed information
    filenames = [Path(f).name for f in outfiles]
    startTimes = [pd.to_datetime(f.split('_')[1]) for f in filenames]
    startTimes = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in startTimes]
    descriptions = [
        'Land Surface Temperature - air temperature (K)'] * len(filenames)
    geometries = [''] * len(filenames)
    df = pd.DataFrame({'geometry': geometries,
                       'startTime': startTimes,
                       'endTime': startTimes,
                       'filename': filenames,
                       'description': descriptions})

    # save as ; separate csv file
    outcsv = output_dir / tile / f'FSTEP_upload_lst-ta_{tile}.csv'
    df.to_csv(outcsv, sep=';', index=False)

    print(f'** Results saved in: {outdir}')
    print(f'** CSV file for FSTEP upload saved in: {outcsv}')

    # Get NDVI data separately
    logger.info('** Getting NDVI data')
    ndvi_files = list(preprocess_dict['SENTINEL2_L2A']['NDVI'].values())
    outdir = output_dir / tile / '006_ndvi'
    outdir.mkdir(parents=True, exist_ok=True)
    outfiles = []
    for f in ndvi_files:
        date = Path(f).name.split('_')[1][:-5]
        dest = outdir / f'NDVI_{date}_{tile}.tif'
        outfiles.append(dest)
        if dest.exists():
            continue
        shutil.copyfile(f, dest)

    # create pandas dataframe with needed information
    filenames = [Path(f).name for f in outfiles]
    startTimes = [pd.to_datetime(f.split('_')[1]) for f in filenames]
    endTimes = [st + pd.Timedelta('1D') for st in startTimes]
    startTimes = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in startTimes]
    endTimes = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in endTimes]

    descriptions = [
        'Normalized difference vegetation index'] * len(filenames)
    geometries = [''] * len(filenames)
    df = pd.DataFrame({'geometry': geometries,
                       'startTime': startTimes,
                       'endTime': endTimes,
                       'filename': filenames,
                       'description': descriptions})

    # save as ; separate csv file
    outcsv = output_dir / tile / f'FSTEP_upload_NDVI_{tile}.csv'
    df.to_csv(outcsv, sep=';', index=False)

    print(f'** Results saved in: {outdir}')
    print(f'** CSV file for FSTEP upload saved in: {outcsv}')

    logger.info('** Computing ET with TSEB-PT model')
    biopar_dict = {var: preprocess_dict[var]
                   for var in SenETDownload.BIOPAR_VARIABLES}
    worldcover_file = preprocess_dict[data_download.name_worldcover]
    s3_dict = preprocess_dict['SENTINEL3_SLSTR_L2_LST']
    outdir_et = output_dir / tile / '007_et'
    outdir_meteo = outdir_et / 'meteo'
    outdir_meteo.mkdir(parents=True, exist_ok=True)

    meteo_cache = {}
    if compute_et_tseb:
        outdir_et.mkdir(parents=True, exist_ok=True)
        outdir_meteo.mkdir(parents=True, exist_ok=True)

        for t in timestamps:
            timestr = t.strftime('%Y%m%dT%H%M%S')
            et_file = outdir_et / f'TSEB-PT_{timestr}_{tile}.vrt'
            if et_file.exists():
                continue

            meteo_paths = compute_meteo_for_tseb(
                t, elev_file, time_zone, era5col, outdir_meteo)
            meteo_cache[t] = meteo_paths

            compute_et(
                tile, t,
                lst_file=lst_dict['LST'][t],
                vza_file=s3_dict['viewZenithAnglesHR'][t],
                lat_file=s3_dict['latHR'][t],
                lon_file=s3_dict['lonHR'][t],
                elev_file=elev_file,
                biopar_dict=biopar_dict,
                worldcover_file=worldcover_file,
                meteo_paths=meteo_paths,
                outdir=outdir_et,
                time_zone=time_zone,
                et_histogram=et_histogram,
            )

    if generate_lstm_like:
        logger.info('** Generating LSTM-like 30 m LST product, LST-Ta and ET')
        outdir_lstm = output_dir / tile / '008_lstm-like'
        outdir_lstm.mkdir(parents=True, exist_ok=True)
        outdir_ta_lstm = output_dir / tile / '008_lstm-ta'
        outdir_ta_lstm.mkdir(parents=True, exist_ok=True)
        outdir_et_lstm = output_dir / tile / '008_lstm-et'
        outdir_et_lstm.mkdir(parents=True, exist_ok=True)
        for t in timestamps:
            timestr = t.strftime('%Y%m%dT%H%M%S')
            lstm_out = outdir_lstm / f'LST-LSTM_{timestr}_{tile}.tif'
            lstm_ta_file = outdir_ta_lstm / f'LST-Ta-LSTM_{timestr}_{tile}.tif'
            lstm_et_file = outdir_et_lstm / f'TSEB-PT_{timestr}_{tile}.vrt'
            if lstm_ta_file.exists() and lstm_et_file.exists():
                continue
            # If an external LST folder is provided, use only those files.
            # Timestamps with no matching external file are skipped entirely.
            if external_lst_folder is not None:
                _ext = _find_external_lst(external_lst_folder, tile, t)
                if _ext is not None:
                    logger.info(
                        f'  Using external LST for {timestr}: {_ext.name}')
                    lstm_out = _ext
                else:
                    logger.warning(
                        f'  No external LST found for {timestr} in '
                        f'{external_lst_folder} — skipping.')
                    continue
            if not lstm_out.exists():
                logger.info(f'  Generating LSTM-like LST: {timestr}')
                generate_lstm_like_lst(
                    sharpened_lst_file=lst_dict['LST'][t],
                    vza_file=s3_dict['viewZenithAnglesHR'][t],
                    outfile=lstm_out,
                )
            if not lstm_ta_file.exists():
                compute_lst_ta(lstm_out, t, elev_file, time_zone,
                               era5col, lstm_ta_file)
            if not lstm_et_file.exists():
                if t in meteo_cache:
                    meteo_paths = meteo_cache[t]
                else:
                    meteo_paths = compute_meteo_for_tseb(
                        t, elev_file, time_zone, era5col, outdir_meteo)

                # All auxiliary inputs are at 20 m but the LSTM-like LST is
                # at 30 m.  _process_tseb_tiled reads every input with the
                # same pixel offsets as T_R1, so a resolution mismatch causes
                # a spatial shift.  Resample everything to the 30 m grid
                # of lstm_out before passing to compute_et.
                inp30 = outdir_lstm / 'inputs_30m'
                inp30.mkdir(parents=True, exist_ok=True)

                # S3 geometry — nearest for angle/coord grids
                vza_30m = resample_to_match(
                    s3_dict['viewZenithAnglesHR'][t], lstm_out,
                    inp30 / f'vza_{timestr}.tif',
                    Resampling.nearest)
                lat_30m = resample_to_match(
                    s3_dict['latHR'][t], lstm_out,
                    inp30 / f'lat_{timestr}.tif',
                    Resampling.bilinear)
                lon_30m = resample_to_match(
                    s3_dict['lonHR'][t], lstm_out,
                    inp30 / f'lon_{timestr}.tif',
                    Resampling.bilinear)

                # Static inputs — computed once, reused across timestamps
                elev_30m = resample_to_match(
                    elev_file, lstm_out,
                    inp30 / 'elev.tif',
                    Resampling.bilinear)
                worldcover_30m = resample_to_match(
                    worldcover_file, lstm_out,
                    inp30 / 'worldcover.tif',
                    Resampling.nearest)

                # Biopar — resample all dates for all variables
                biopar_dict_30m = {}
                for var, date_paths in biopar_dict.items():
                    biopar_dict_30m[var] = {}
                    for bdate, bpath in date_paths.items():
                        bdst = inp30 / (
                            f'{var}_{bdate.strftime("%Y%m%d")}.tif')
                        biopar_dict_30m[var][bdate] = resample_to_match(
                            bpath, lstm_out, bdst, Resampling.bilinear)

                # Meteo — resample all fields to 30 m
                meteo_30m = {
                    k: resample_to_match(
                        v, lstm_out,
                        inp30 / f'meteo_{timestr}_{k}.tif',
                        Resampling.bilinear)
                    for k, v in meteo_paths.items()
                }

                compute_et(
                    tile, t,
                    lst_file=lstm_out,
                    vza_file=vza_30m,
                    lat_file=lat_30m,
                    lon_file=lon_30m,
                    elev_file=elev_30m,
                    biopar_dict=biopar_dict_30m,
                    worldcover_file=worldcover_30m,
                    meteo_paths=meteo_30m,
                    outdir=outdir_et_lstm,
                    time_zone=time_zone,
                    biopar_cache_dir=outdir_lstm / 'biopar',
                    et_histogram=et_histogram,
                )

        # Clean up LSTM-like intermediates
        inp30_path = output_dir / tile / '008_lstm-like' / 'inputs_30m'
        if inp30_path.exists():
            logger.info('Cleaning up LSTM-like intermediate files...')
            shutil.rmtree(inp30_path)

    logger.info('** All done!')


if __name__ == "__main__":

    # NOTE that in order to avoid processing issues, the temporal extent
    # should be limited to a maximum of 6 months.

    tiles = ['35VMF']#31UFS , '35VMF', '32UPC
    temporal_blocks = [
        ['2024-01-01', '2024-04-30'],
        ['2024-05-01', '2024-09-30'],
        ['2024-10-01', '2024-12-31'],
    ]
    # Use a common root output directory; per-tile folders are created inside.
    output_dir = Path('/vitodata/CHILL_Y/OPENEO/2024')
    era5_tiled_folder = Path('/vitodata/CHILL_Y/data/ERA5')
    time_zone = 0


    # NOTE: if residual correction is activated,
    # then the result of the sharpening
    # algorithm will display some clear blocky artifacts,
    # as each low resolution pixel is corrected with the same residual value.
    residual_correction = False
    # parameters for bias + directional correction
    # if None, no correction is applied
    corr_parameters = None
    # set to True to also produce a 30 m LSTM-like LST product
    # (VZA > 30° masked, LST resampled via radiance space)
    generate_lstm_like = True


    # set to False to skip the TSEB-PT ET computation entirely
    compute_et_tseb = True
    # — useful for a quick sanity check of the TSEB-PT output
    # set to True to save a PNG histogram of ET_day next to each VRT    
    et_histogram = True
    # Minimum fraction of valid S3 pixels required to run sharpening.
    # Scenes below this threshold are skipped (e.g. 0.05 = 5%).
    # Set to 0.0 to keep the original behaviour (skip only fully-empty scenes).
    min_valid_s3_fraction = 0.05
    # If True, the sharpened LST is masked to only pixels where the original
    # S3 observation was valid (no extrapolation outside S3 coverage).
    mask_to_s3_coverage = False

    # -------------------------------------------------------------------------
    # INTERNAL USE ONLY: folder containing pre-existing LST GeoTIFFs to use
    # instead of the sharpened S3 LST for the LSTM-like LST-Ta and ET products.
    # Files are matched to S3 timestamps by date (YYYYMMDD in the filename).
    # Set to None to use the standard sharpened LST (default behaviour).
    # -------------------------------------------------------------------------
    external_lst_folder = Path('/vitodata/CHILL_Y/LSTMLikeDataset/')
    # external_lst_folder = Path('/vitodata/CHILL_Y/LSTMLikeDataset/')

    # Number of months per BIOPAR OpenEO job. Splitting into smaller chunks
    # reduces per-job memory so the default executor settings can be used.
    # Set to 1 for monthly jobs (recommended), or 0 to use a single job
    # covering the full temporal extent (original behaviour).
    biopar_chunk_months = 1

    # Number of months per Sentinel-2 OpenEO job. Same rationale as
    # biopar_chunk_months above. Set to 1 for monthly jobs (recommended),
    # or 0 to use a single job covering the full temporal extent.
    s2_chunk_months = 1

    # Maximum number of concurrent OpenEO jobs submitted via the
    # MultiBackendJobManager (used for chunked S2 and BIOPAR downloads).
    # Set to match your account's concurrent job limit.
    max_concurrent_jobs = 4

    for tile in tiles:
        for temporal_extent in temporal_blocks:
            logger.info(
                f'** Processing tile {tile} for block '
                f'{temporal_extent[0]} -> {temporal_extent[1]}')
            main(tile, temporal_extent, time_zone, output_dir,
                 era5_tiled_folder,
                 residual_correction, corr_parameters,
                 generate_lstm_like=generate_lstm_like,
                 et_histogram=et_histogram,
                 compute_et_tseb=compute_et_tseb,
                 min_valid_s3_fraction=min_valid_s3_fraction,
                 mask_to_s3_coverage=mask_to_s3_coverage,
                 external_lst_folder=external_lst_folder,
                 biopar_chunk_months=biopar_chunk_months,
                 s2_chunk_months=s2_chunk_months,
                 max_concurrent_jobs=max_concurrent_jobs)
