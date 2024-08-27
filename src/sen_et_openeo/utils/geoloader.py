import rasterio
import numpy as np
from pathlib import Path
from osgeo import gdal
from osgeo import osr
import netCDF4
import datetime

from satio.geoloader import ParallelLoader
from satio.utils.geotiff import (get_rasterio_profile,
                                 write_geotiff_tags)

from sen_et_openeo.utils.timedate import _bracketing_dates


S2_BANDS = ['S2-B02', 'S2-B03', 'S2-B04', 'S2-B05',
            'S2-B06', 'S2-B07', 'S2-B08', 'S2-B8A',
            'S2-B11', 'S2-B12']

S2_BIOPARS = ['S2-fapar_8', 'S2-lai', 'S2-fcover',
              'S2-ccc', 'S2-cwc', 'S2-fgreenveg']

S2_LEAFREFL_BANDS = ['S2-refl_vis', 'S2-trans_vis',
                     'S2-refl_nir', 'S2-trans_nir']

S2_VEGSTRUCT_BANDS = ['veg_inclination_distribution',
                      'veg_leaf_width',
                      'veg_height_width_ratio']

S3_GEOBANDS = ['S3-lat', 'S3-lon',
               'S3-latHR', 'S3-lonHR',
               'S3-vza', 'S3-sza',
               'S3-vzaHR', 'S3-szaHR',
               'S3-inc']

OUTPUT_SCALING = {
    'S2-refl': {'scale': 10000,
                'dtype': np.uint16,
                'nodata': 0},
    'S2-leafrefl': {'scale': 10000,
                    'dtype': np.uint16,
                    'nodata': 0},
    'S2-fapar_8': {'scale': 100,
                   'dtype': np.uint8,
                   'nodata': 0},
    'S2-lai': {'scale': 10,
               'dtype': np.uint8,
               'nodata': 0},
    'S2-fcover': {'scale': 100,
                  'dtype': np.uint8,
                  'nodata': 0},
    'S2-ccc': {'scale': 100,
               'dtype': np.uint32,
               'nodata': 0},
    'S2-cwc': {'scale': 1000,
               'dtype': np.uint16,
               'nodata': 0},
    'S2-fgreenveg': {'scale': 100,
                     'dtype': np.uint8,
                     'nodata': 0},
    'veg_height': {'scale': 10,
                   'dtype': np.uint16,
                   'nodata': np.iinfo(np.uint16).max},
    'veg_struct': {'scale': 100,
                   'dtype': np.uint8,
                   'nodata': 255},
    'igbp_classification': {'scale': None,
                            'dtype': np.uint8,
                            'nodata': 255},
    'roughness_length': {'scale': 10000,
                         'dtype': np.uint32,
                         'nodata': np.iinfo(np.uint32).max},
    'zero_plane_displacement': {'scale': 100,
                                'dtype': np.uint16,
                                'nodata': np.iinfo(np.uint16).max},
    'DEM-alt-20m': {'scale': None,
                    'dtype': np.int16,
                    'nodata': 11000},
    'DEM-slo-20m': {'scale': None,
                    'dtype': np.float32,
                    'nodata': -9999},
    'DEM-asp-20m': {'scale': None,
                    'dtype': np.uint8,
                    'nodata': 255},
    'S3-LST': {'scale': 100,
               'dtype': np.uint16,
               'nodata': 0},
    'S3-LSTHR': {'scale': 100,
                 'dtype': np.uint16,
                 'nodata': 0},
    'S3-LSTHR-nocor': {'scale': None,
                       'dtype': np.float32,
                       'nodata': None},
    'S3-geo': {'scale': None,
               'dtype': np.float32,
               'nodata': -9999},
    'S3-mask': {'scale': None,
                'dtype': np.uint8,
                'nodata': 255},
    'S3-CV': {'scale': None,
              'dtype': np.float32,
              'nodata': -9999},
    'METEO-air_pressure': {'scale': 100,
                           'dtype': np.uint16,
                           'nodata': np.iinfo(np.uint16).max},
    'METEO-air_temperature': {'scale': 100,
                              'dtype': np.uint16,
                              'nodata': 0},
    'METEO-average_daily_solar_irradiance': {'scale': 10,
                                             'dtype': np.uint16,
                                             'nodata': np.iinfo(np.uint16).max},
    'METEO-clear_sky_solar_radiation': {'scale': 10,
                                        'dtype': np.uint16,
                                        'nodata': np.iinfo(np.uint16).max},
    'METEO-t2m': {'scale': 100,
                  'dtype': np.uint16,
                  'nodata': 0},
    'METEO-vapour_pressure': {'scale': 100,
                              'dtype': np.uint16,
                              'nodata': np.iinfo(np.uint16).max},
    'METEO-wind_speed': {'scale': 1000,
                         'dtype': np.uint16,
                         'nodata': np.iinfo(np.uint16).max},
    'lst-ta': {'scale': 100,
               'dtype': np.int16,
               'nodata': np.iinfo(np.int16).max},
    'et': {'scale': 100,
           'dtype': np.uint16,
           'nodata': np.iinfo(np.uint16).max},
}

DEM_FEATURES = ['DEM-alt-20m', 'DEM-asp-20m', 'DEM-slo-20m']


def GetExtent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    ''' Rep roject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def get_GeoInfo(rasterfile):
    '''
    Function returns the geo information of a rasterfile

    :param rasterfile: input rasterfile
    :return: geo_info dictionary with geo information
    '''
    ds = gdal.Open(rasterfile)
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = GetExtent(gt, cols, rows)
    resx = gt[1]
    resy = gt[5]

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    tgt_srs = src_srs.CloneGeogCS()
    geo_ext = np.array(ReprojectCoords(ext, src_srs, tgt_srs))

    geo_info = dict()

    geo_info['extent'] = ext
    geo_info['x_min'] = min(np.array(ext)[:, 0])
    geo_info['x_max'] = max(np.array(ext)[:, 0])
    geo_info['y_min'] = min(np.array(ext)[:, 1])
    geo_info['y_max'] = max(np.array(ext)[:, 1])
    geo_info['Xdim'] = ds.RasterXSize
    geo_info['Ydim'] = ds.RasterYSize
    geo_info['srs'] = src_srs.ExportToProj4()
    geo_info['Xres'] = resx
    geo_info['Yres'] = resy

    geo_info['geo_ext'] = geo_ext

    return geo_info


def clip2rastExt(source_file, extent_file, outfile):
    '''
    function to clip a .tiff raster using the extent of another .tiff raster
    :param source_file: the raster file to be clipped
    :param extent_file: the raster file defining the extent
    :param outfile: the output geotif file to create
    :return:
    '''

    ds = gdal.Open(source_file)
    info = get_GeoInfo(extent_file)
    ext = [info['x_min'], info['y_max'], info['x_max'], info['y_min']]
    dso = gdal.Translate(outfile, ds, projWin=ext)
    dso = None
    ds = None


def get_band_metadata(band):

    if band in S2_BANDS:
        outinfo = OUTPUT_SCALING.get('S2-refl')
    elif band in S2_LEAFREFL_BANDS:
        outinfo = OUTPUT_SCALING.get('S2-leafrefl')
    elif band in S2_VEGSTRUCT_BANDS:
        outinfo = OUTPUT_SCALING.get('veg_struct')
    elif band in S3_GEOBANDS:
        outinfo = OUTPUT_SCALING.get('S3-geo')
    else:
        outinfo = OUTPUT_SCALING.get(band, None)

    if outinfo is not None:
        scale = outinfo['scale']
        dtype = outinfo['dtype']
        nodata = outinfo['nodata']
    else:
        scale = None
        dtype = None
        nodata = 0

    return scale, dtype, nodata


def readraster(infile, bandnr=1):

    # read data
    with rasterio.open(infile) as src:
        array = src.read(bandnr)
        nodata = src.nodata
        scale = src.scales[bandnr-1]
        offset = src.offsets[bandnr-1]

    # scale and apply nodata value
    if nodata is not None:
        array = array.astype(np.float32)
        array[array == nodata] = np.nan
    array = (array * scale) + offset

    return array


def getrasterinfo(infile):
    with rasterio.open(infile) as src:
        proj = src.crs
        epsg = str(proj)[5:]
        gt = src.transform
        sizeX, sizeY = src.res
        bounds = src.bounds
        bands = src.count
        if 'bands' in src.tags().keys():
            bandnames = eval(src.tags()['bands'])
        else:
            bandnames = []

    return epsg, bounds, gt, sizeX, sizeY, bands, bandnames


def writeraster(array, outfile, epsg, bounds,
                bandname=None):

    # derive band name from filename if needed
    if bandname is None:
        basename = Path(outfile).name
        if len(basename.split('_')) == 4:
            # dealing with timeseries
            bandname = basename.split('_')[-2]
        else:
            # dealing with feature
            bandname = basename.split('_')[-1].split('.')[0]

    # get metadata
    scale, dtype, nodata = get_band_metadata(bandname)

    # apply scaling and nodata value
    if scale is not None:
        array = array * scale
        scales = [1/scale]
    else:
        scales = None
    if nodata is not None:
        array[np.isnan(array)] = nodata
    if dtype is not None:
        array = array.astype(dtype)

    # now write the array
    profile = get_rasterio_profile(array, bounds, epsg)
    bands_tags = [{'band_name': bandname}]
    write_geotiff_tags(array, profile, outfile,
                       bands_tags=bands_tags,
                       nodata=nodata, scales=scales)


def saveImgMem(data, geotransform, proj, outPath,
               noDataValue=None):
    '''Save the data to memory using GDAL'''

    memDriver = gdal.GetDriverByName("MEM")
    shape = data.shape
    if len(shape) > 2:
        ds = memDriver.Create("MEM", shape[2], shape[1], shape[0],
                              gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[0]):
            ds.GetRasterBand(i+1).WriteArray(data[i, ...])
    else:
        ds = memDriver.Create("MEM", shape[1], shape[0], 1,
                              gdal.GDT_Float32)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)

    # Save to file if required
    if outPath == "MEM":
        if noDataValue is None:
            noDataValue = np.nan
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)

    return ds


class S3ParallelLoader(ParallelLoader):

    def load(self, collection, bands, resolution):
        if not isinstance(bands, (list, tuple)):
            raise TypeError("'bands' should be a list/tuple of bands. "
                            f"Its type is: {type(bands)}")

        arrs_list = self._load_raw(collection, bands, resolution)
        products = collection.products
        timestamps = collection.timestamps
        bands = list(bands)
        bounds = list(collection.bounds)
        epsg = collection.epsg

        xds_dict = {band: self._arr_to_xarr(arrs_list[i],
                                            bounds,
                                            timestamps,
                                            name=band)
                    for i, band in enumerate(bands)}

        xds_dict.update({'epsg': epsg,
                         'bounds': bounds,
                         'products': products,
                         'bands': bands})

        return xds_dict


def _getECMWFTempInterpData(ncfile, var_name, before_I, after_I, frac):

    ds = gdal.Open('NETCDF:"'+ncfile+'":'+var_name)
    if ds is None:
        raise RuntimeError(
            "Variable %s does not exist in file %s." % (var_name, ncfile))

    # Get some metadata
    scale = ds.GetRasterBand(before_I+1).GetScale()
    offset = ds.GetRasterBand(before_I+1).GetOffset()
    no_data_value = ds.GetRasterBand(before_I+1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()

    # Read the right time layers
    try:
        data_before = ds.GetRasterBand(before_I+1).ReadAsArray()
        data_before = (data_before.astype(float) * scale) + offset
        data_before[data_before == no_data_value] = np.nan
        data_after = ds.GetRasterBand(after_I+1).ReadAsArray()
        data_after = (data_after.astype(float) * scale) + offset
        data_after[data_after == no_data_value] = np.nan
    except AttributeError:
        ds = None
        raise RuntimeError(
            "ECMWF file does not contain data for the requested date.")

    # Perform temporal interpolation
    data = data_before*frac + data_after*(1.0-frac)

    return data, gt, proj


def get_timing(ncfile, date_time):

    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ncfile, 'r')
    time = fid.variables['time']
    dates = netCDF4.num2date(time[:], time.units, time.calendar)
    del fid

    timing, _, _ = _bracketing_dates(dates, date_time)
    return timing


def _getECMWFIntegratedData(ncfile, var_name, date_time, time_window=24,):

    # Get the time right before date_time,
    # to use it as integrated baseline
    datestr = date_time.strftime(format='%Y-%m-%d')
    ncfile_0 = str(Path(ncfile).parent /
                   f'{datestr}.nc')
    timing_0 = get_timing(ncfile_0, date_time)

    # Get the time right after the temporal window set
    end_date_time = date_time + datetime.timedelta(hours=time_window)
    datestr = end_date_time.strftime(format='%Y-%m-%d')
    ncfile_1 = str(Path(ncfile).parent /
                   f'{datestr}.nc')
    timing_1 = get_timing(ncfile_1, end_date_time)

    ds = gdal.Open('NETCDF:"'+ncfile_0+'":'+var_name)
    if ds is None:
        raise RuntimeError(
            "Variable %s does not exist in file %s." % (var_name, ncfile))
    # Get some metadata
    scale = ds.GetRasterBand(timing_0+1).GetScale()
    offset = ds.GetRasterBand(timing_0+1).GetOffset()
    no_data_value = ds.GetRasterBand(timing_0+1).GetNoDataValue()
    # Report geolocation of the top-left pixel of rectangle
    gt = ds.GetGeoTransform()
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()

    # Forecasts of ERA5 the accumulations are since the
    # previous post processing (archiving)
    data_ref = 0

    # Initialize output variable
    cummulated_value = 0.

    # determine whether we have to read from 1 or 2 files
    if ncfile_0 == ncfile_1:
        # just one file: read the range as computed
        for date_i in range(timing_0+1, timing_1+1):
            # Read the right time layers
            data = ds.GetRasterBand(date_i+1).ReadAsArray()
            data = (data.astype(float) * scale) + offset
            data[data == no_data_value] = 0
            # The time step value is the difference between
            # the actual timestep value and the previous value
            cummulated_value += (data - data_ref)

    else:
        # first read first file till end
        for date_i in range(timing_0+1, 24):
            # Read the right time layers
            data = ds.GetRasterBand(date_i+1).ReadAsArray()
            data = (data.astype(float) * scale) + offset
            data[data == no_data_value] = 0
            cummulated_value += (data - data_ref)

        # now read the second file from start
        ds = gdal.Open('NETCDF:"'+ncfile_1+'":'+var_name)
        for date_i in range(0, timing_1+1):
            # Read the right time layers
            data = ds.GetRasterBand(date_i+1).ReadAsArray()
            data = (data.astype(float) * scale) + offset
            data[data == no_data_value] = 0
            cummulated_value += (data - data_ref)

    # Convert to average W m^-2
    cummulated_value = cummulated_value / (time_window * 3600.)

    return cummulated_value, gt, proj
