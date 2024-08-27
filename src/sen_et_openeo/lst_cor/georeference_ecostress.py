import os

import h5py
import numpy as np
import pyproj
from osgeo import gdal, gdal_array, osr
from pyresample import geometry as geom
from pyresample import kd_tree as kdt

"""
for more information about the script: https://lpdaac.usgs.gov/resources/e-learning/working-ecostress-evapotranspiration-data/
"""


def search_attribute_ID_in_h5py(file, attribute_name):
    # Open Geo File
    objs = []
    file.visit(objs.append)

    # Search for lat/lon SDS inside data file
    attribute_SD = [str(obj) for obj in objs if isinstance(
        file[obj], h5py.Dataset) and attribute_name in obj]

    return attribute_SD[0]


def get_latitude_and_longitude(g):
    # Open Geo File
    geo_objs = []
    g.visit(geo_objs.append)

    # Search for lat/lon SDS inside data file
    latSD = [str(obj) for obj in geo_objs if isinstance(
        g[obj], h5py.Dataset) and '/latitude' in obj]
    lonSD = [str(obj) for obj in geo_objs if isinstance(
        g[obj], h5py.Dataset) and '/longitude' in obj]

    # Open SDS as arrays
    latitude = g[latSD[0]][()].astype(float)
    longitude = g[lonSD[0]][()].astype(float)
    return latitude, longitude


def get_swath(lat, lon):
    # Set swath definition from lat/lon arrays
    swathDef = geom.SwathDefinition(lons=lon, lats=lat)
    # Define the lat/lon for the middle of the swath
    mid = [int(lat.shape[1] / 2) - 1, int(lat.shape[0] / 2) - 1]
    midLat, midLon = lat[mid[0]][mid[1]], lon[mid[0]][mid[1]]

    # Define AEQD projection centered at swath center
    epsgConvert = pyproj.Proj(
        "+proj=aeqd +lat_0={} +lon_0={}".format(midLat, midLon))

    # Use info from AEQD projection bbox to calculate output cols/rows/pixel size
    llLon, llLat = epsgConvert(np.min(lon), np.min(lat), inverse=False)
    urLon, urLat = epsgConvert(np.max(lon), np.max(lat), inverse=False)
    areaExtent = (llLon, llLat, urLon, urLat)
    cols = int(round((areaExtent[2] - areaExtent[0]) / 70))  # 70 m pixel size
    rows = int(round((areaExtent[3] - areaExtent[1]) / 70))

    # Set parameters for the Geographic projection
    epsg, proj, pName = '4326', 'longlat', 'Geographic'

    # Define bounding box of swath
    llLon, llLat, urLon, urLat = np.min(
        lon), np.min(lat), np.max(lon), np.max(lat)
    areaExtent = (llLon, llLat, urLon, urLat)

    # Create a CRS (Coordinate Reference System) for the specified EPSG code
    projDict = pyproj.CRS("epsg:4326")
    # Create an AreaDefinition for the specified projection and dimensions
    areaDef = geom.AreaDefinition(
        epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Square pixels and calculate output cols/rows
    # Determine the minimum pixel size (square pixels)
    ps = np.min([areaDef.pixel_size_x, areaDef.pixel_size_y])
    # Calculate the number of columns and rows based on pixel size and area extent
    cols = int(round((areaExtent[2] - areaExtent[0]) / ps))
    rows = int(round((areaExtent[3] - areaExtent[1]) / ps))

    # Set up a new Geographic area definition with the refined cols/rows
    areaDef = geom.AreaDefinition(
        epsg, pName, proj, projDict, cols, rows, areaExtent)

    # Define the geotransform
    gt = [areaDef.area_extent[0], ps, 0, areaDef.area_extent[3], 0, -ps]

    return cols, rows, swathDef, areaDef, epsg, gt


def georeference_and_create_geotiffs(bands, out_dir, gt, epsg, filename, fill_values, areaDef, index, outdex, indexArr):
    lst_geo = bands["LST"]
    nr_bands = len(bands)

    # Set up output name
    outName = os.path.join(out_dir, filename)

    # Get driver, specify dimensions, define and set output geotransform
    height, width = areaDef.shape
    driv = gdal.GetDriverByName('GTiff')
    dataType = gdal_array.NumericTypeCodeToGDALTypeCode(lst_geo.dtype)
    # TODO make sure that outName directory exists
    d = driv.Create(outName, width, height, nr_bands, dataType)
    d.SetGeoTransform(gt)
    # Create and set output projection, write output array data
    # Define target SRS
    srs = osr.SpatialReference()
    succes = 0 == srs.ImportFromEPSG(int(epsg))
    if succes:
        # TODO: check if a projection is set!!!!
        d.SetProjection(srs.ExportToWkt())
    else:
        d.SetProjection(
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')

    # Write array to band
    for band_nr, band_name in enumerate(bands, start=1):
        georeferenced_band = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, bands[band_name], index, outdex,
                                                                indexArr,
                                                                fill_value=fill_values[band_name])

        # TODO find a way to convert all fill values to the same value
        fv = fill_values[band_name]
        band = d.GetRasterBand(band_nr)
        band.SetDescription(band_name)
        band.WriteArray(georeferenced_band)

        # Define fill value if it exists, if not, set to mask fill value
        fv = band.SetNoDataValue(float(1000))
        # if fv is not None and fv != 'NaN':
        #     band.SetNoDataValue(float(fv))
        # else:
        #     try:
        #         band.SetNoDataValue(bands[band].fill_value)
        #     except:
        #         pass
        band.FlushCache()
        band = None
        georeferenced_band = None
    d = None
    return


def retrieve_h5py_array(file_path, attribute_name):
    f = h5py.File(file_path, 'r')  # Read in ECOSTRESS HDF5 file
    fill_value = None
    if "GEO" in file_path:
        if f["L1GEOMetadata"]['OrbitCorrectionPerformed'][()].decode() == 'False':
            return False, False

    # attribute_array = np.array(f[attribute_name]).astype(float)
    attribute_ID = search_attribute_ID_in_h5py(f, attribute_name)

    # Open SDS as arrays
    attribute_array = f[attribute_ID][()].astype(float)

    try:
        fill_value = f[attribute_ID].attrs['fill_value']
    except:
        pass
    # Rescale
    try:
        scale_factor = f[attribute_ID].attrs['scale_factor']
        attribute_array = attribute_array * scale_factor

    except:
        pass

    f.close()
    del f
    return attribute_array, fill_value


def nearest_neighbour_sampling(bands_dict, areaDef, index, outdex, indexArr, fv_dict):
    georeferenced_bands = {}
    for band_name, band in bands_dict.items():
        georeferenced_bands[band_name] = kdt.get_sample_from_neighbour_info('nn', areaDef.shape, band, index, outdex,
                                                                            indexArr,
                                                                            fill_value=fv_dict[band_name])

    return georeferenced_bands


def ecostress_hp5y_attributes(lst_path, geo_path, cloud_path, include_viewing_geometry):
    # endregion

    attribute_dictionary = {}
    fill_value_dictionary = {}
    lste_attributes = ['LST']

    if include_viewing_geometry:
        geo_attributes = ['view_zenith', 'view_azimuth',
                          'solar_zenith', 'solar_azimuth', 'land_fraction']
    else:
        geo_attributes = ['land_fraction']

    cloud_attributes = ['CloudMask']

    for attribute in lste_attributes:
        attribute_array, fill_value = retrieve_h5py_array(lst_path, attribute)
        attribute_dictionary[attribute] = attribute_array
        fill_value_dictionary[attribute] = fill_value

    for attribute in geo_attributes:
        attribute_array, fill_value = retrieve_h5py_array(geo_path, attribute)
        if attribute_array is False:
            return False, False

        attribute_dictionary[attribute] = attribute_array
        fill_value_dictionary[attribute] = fill_value

    for attribute in cloud_attributes:
        attribute_array, fill_value = retrieve_h5py_array(
            cloud_path, attribute)
        attribute_dictionary[attribute] = attribute_array
        fill_value_dictionary[attribute] = fill_value

    return attribute_dictionary, fill_value_dictionary


def georeference_ecostress(lst_path, geo_path, cloud_path, output_path, include_viewing_geometry=False):
    # Get all attribute arrays
    attribute_dictionary, fill_value_dictionary = ecostress_hp5y_attributes(lst_path, geo_path, cloud_path,
                                                                            include_viewing_geometry)

    if not attribute_dictionary:
        return False

    # region output folder
    # Split the path into directory and filename
    out_dir, filename = os.path.split(output_path)
    # Check if out_dir exists, create it if not
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # endregion

    # region GEOLOCATION DATA
    f_geo = h5py.File(geo_path, 'r')
    lat, lon = get_latitude_and_longitude(f_geo)
    f_geo.close()
    del f_geo

    # endregion

    # region Geographic projection

    # Get information about the swath, columns, rows, and area extent
    cols, rows, swathDef, areaDef, epsg, gt = get_swath(lat, lon)

    # Get arrays with information about the nearest neighbor to each grid point
    index, outdex, indexArr, distArr = kdt.get_neighbour_info(
        swathDef, areaDef, 210, neighbours=1)

    georeference_and_create_geotiffs(attribute_dictionary, out_dir, gt, epsg, filename, fill_value_dictionary, areaDef,
                                     index, outdex, indexArr)
    return True
