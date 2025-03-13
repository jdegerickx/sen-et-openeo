import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.profiles import Profile


class DefaultProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""

    defaults = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'deflate',
        'dtype': 'float32'
    }
    

def get_blocksize(val):
    """
    Blocksize needs to be a multiple of 16
    """
    if val % 16 == 0:
        return val
    else:
        return (val // 16) * 16
    
    
def get_rasterio_profile(arr,
                         bounds,
                         epsg,
                         blockxsize=None,
                         blockysize=None,
                         **params):

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=0)

    base_profile = DefaultProfile()
    shape = arr.shape

    count, height, width = shape

    if blockxsize is None:
        blockxsize = get_blocksize(width)

    if blockysize is None:
        blockysize = get_blocksize(height)

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rasterio.transform.from_bounds(*bounds,
                                                 width=width,
                                                 height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=arr.dtype,
        crs=crs,
        count=count)

    base_profile.update(**params)

    return base_profile

def write_geotiff(arr,
                  profile,
                  filename,
                  band_names=None,
                  colormap=None,
                  nodata=None):

    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr)
        if band_names is not None:
            dst.update_tags(bands=band_names)
            for i, b in enumerate(band_names):
                dst.update_tags(i + 1, band_name=b)

        if colormap is not None:
            dst.write_colormap(
                1, colormap)
            
            
def write_geotiff_tags(arr,
                       profile,
                       filename,
                       colormap=None,
                       nodata=None,
                       tags=None,
                       bands_tags=None,
                       scales=None,
                       offsets=None):
    """
    Utility to write an array to a geotiff

    Parameters
    ----------
    arr : np.array

    profile : rasterio.profile

    colormap : dict
        Colormap for first band

    nodata : int

    tags : dict
        dataset metadata tags to write

    bands_tags : List[Dict]
        list of dictionaries to write to the corresponding bands

    scales : List
        List of scale values to write to the dataset. List length should
        be arr.shape[0]

    offsets : List
        List of offset values to write to the dataset. List length should
        be arr.shape[0]make
    """
    bands_tags = bands_tags if bands_tags is not None else []

    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr)

        if tags is not None:
            dst.update_tags(**tags)

        for i, bt in enumerate(bands_tags):
            dst.update_tags(i + 1, **bt)

        if colormap is not None:
            dst.write_colormap(
                1, colormap)

        if scales is not None:
            dst.scales = scales

        if offsets is not None:
            dst.offsets = offsets