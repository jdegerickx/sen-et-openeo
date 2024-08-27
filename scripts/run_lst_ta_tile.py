
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
import pandas as pd
from loguru import logger
import rasterio
import numpy as np
from rasterio.enums import Resampling
import shutil

from sen_et_openeo.data_download import SenETDownload
from sen_et_openeo.era5 import (ERA5Collection,
                                ERA5TimeSeriesProcessor,
                                get_default_rsi_meta)


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

    scale = 0.01
    nodata = 0

    # Get the LST data
    with rasterio.open(lst_file, 'r') as src:
        output_profile = src.profile.copy()
        lst_data = src.read(1)

    lst_data = lst_data * scale
    lst_data[lst_data == nodata] = np.nan

    # Get the air temperature data from ERA5
    meteo_settings = {'bands': ['t2m']}
    meteo_rsi_meta = get_default_rsi_meta().get('ERA5')
    elev = None

    meteo_ts = ERA5TimeSeriesProcessor([time],
                                       elev,
                                       elev_file,
                                       time_zone,
                                       era5col,
                                       meteo_settings,
                                       rsi_meta=meteo_rsi_meta,
                                       ).compute_ts()

    lst_ta = lst_data - np.squeeze(meteo_ts.data)

    newnodata = -999
    lst_ta = lst_ta / scale
    lst_ta[np.isnan(lst_ta)] = newnodata
    lst_ta = lst_ta.astype(np.int16)

    output_profile.update(dtype=rasterio.int16, nodata=newnodata)

    # write result to file
    with rasterio.open(outfile, 'w', **output_profile) as dst:
        dst.write(lst_ta, 1)
        dst.scales = [scale]
        dst.descriptions = ['LST-Ta']
        dst.units = ['K']
        # Generate overviews
        dst.build_overviews(
            (4, 8, 16), Resampling.average)


def main(tile, temporal_extent, time_zone, output_dir, era5_tiled_folder,
         residual_correction=False, corr_parameters=None,
         parallel_jobs=True, delete_tmp_data=False):

    logger.info('** Downloading data from OpenEO')
    data_download = SenETDownload(tile, temporal_extent)
    data_download.download(output_dir, output_format='gtiff',
                           parallel=parallel_jobs)

    logger.info('** Preprocessing data')
    preprocess_dict = data_download.preprocess(
        output_dir, delete_unrequired_data=delete_tmp_data)

    logger.info('** Running LST sharpening algorithm')
    sharpening_dict = data_download.sharpening(output_dir, residual_correction)

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

    logger.info('** All done!')


if __name__ == "__main__":

    tiles = ['34HBH']
    temporal_extent = ["2019-07-01", "2019-07-20"]
    output_dir = Path('/vitodata/aries/s-africa_test')
    era5_tiled_folder = Path('/data/beresilient/ERA5')
    time_zone = 0

    # # MALI
    # tiles = ['30QVD', '30QWD']
    # temporal_extent = ['2023-06-01', '2024-06-30']
    # output_dir = Path('/vitodata/aries/Mali')
    # era5_tiled_folder = Path('/vitodata/aries/data/ERA5')
    # time_zone = 0

    # # ZAMBIA
    # tiles = ['35LPD']
    # temporal_extent = ['2023-09-01', '2024-08-15']
    # output_dir = Path('/vitodata/aries/Zambia_2')
    # era5_tiled_folder = Path('/vitodata/aries/data/ERA5')
    # time_zone = 2

    # NOTE: if residual correction is activated,
    # then the result of the sharpening
    # algorithm will display some clear blocky artifacts,
    # as each low resolution pixel is corrected with the same residual value.
    residual_correction = False
    # parameters for bias + directional correction
    # if None, no correction is applied
    corr_parameters = {"cross_cal_gain": 1.110024074017716,
                       "cross_cal_offset": -33.008132822189395,
                       "vinnikov_parameter": -7.547175199010964}

    for tile in tiles:
        main(tile, temporal_extent, time_zone, output_dir, era5_tiled_folder,
             residual_correction, corr_parameters)
