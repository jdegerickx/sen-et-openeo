"""
Series of functions to downsample uint16 rasters
or upsample a uint8 raster (SCL mask)
"""
import warnings

import numpy as np
from numba import njit, uint16  # NOQA
import skimage
from skimage.transform import rescale, resize


def upsample(mask):

    arr_type = eval(mask.dtype.name)

    @njit(arr_type[:, :](arr_type[:, :]))
    def _upsample(mask):
        new_shape = (mask.shape[0] * 2, mask.shape[1] * 2)
        new_mask = np.zeros(new_shape, dtype=mask.dtype)
        for i in range(0, new_shape[0], 2):
            for j in range(0, new_shape[1], 2):
                new_mask[i:i+2, j:j+2] = mask[i//2, j//2]
        mask = None
        return new_mask

    @njit(arr_type[:, :, :](arr_type[:, :, :]))
    def _upsample_multichannel(mask):

        new_shape = (mask.shape[0], mask.shape[1] * 2, mask.shape[2] * 2)
        new_mask = np.zeros(new_shape, dtype=mask.dtype)
        for i in range(new_shape[0]):
            new_mask[i, ...] = _upsample(mask[i, ...])
        mask = None
        return new_mask

    if len(mask.shape) == 2:
        return _upsample(mask)
    elif len(mask.shape) == 3:
        return _upsample_multichannel(mask)
    else:
        raise ValueError(f"Input shape {mask.shape} not recognized.")

@njit(uint16[:, :](uint16[:, :]))
def block_downsample_band(band):
    new_shape = (band.shape[0] // 2, band.shape[1] // 2)
    new_band = np.zeros(new_shape, dtype=np.uint16)
    for i in range(0, band.shape[0], 2):
        for j in range(0, band.shape[1], 2):
            new_band[i//2, j//2] = np.uint16(np.mean(band[i:i+2, j:j+2]))
    band = None
    return new_band


@njit(uint16[:, :, :](uint16[:, :, :]))
def block_downsample_band3d(band):
    new_shape = (band.shape[0],
                 band.shape[1] // 2,
                 band.shape[2] // 2)

    new_band = np.zeros(new_shape, dtype=np.uint16)

    for i in range(new_shape[0]):
        new_band[i, ...] = block_downsample_band(band[i, :, :])
    band = None
    return new_band


def downsample_n(arr, n):

    for i in range(n):
        arr = block_downsample_band3d(arr)

    return arr


def imresize(img, scaling=0, order=1, shape=None, anti_aliasing=True):
    """
    Perform resampling of an image, either performing block average when
    downsampling, or using skimage rescale when upsampling
    """

    if shape is not None:
        with warnings.catch_warnings():  # ignore skimage warnings
            warnings.simplefilter("ignore")
            img = resize(img,
                         shape,
                         order=order,
                         anti_aliasing=anti_aliasing,
                         preserve_range=True).astype(np.float32)
            return img

    if (scaling == 0) or ~np.isfinite(scaling):
        ValueError('scaling value {} is not valid.'.format(scaling))

    integer_inverse = (1 / scaling) % 1 == 0

    with warnings.catch_warnings():  # ignore skimage warnings
        warnings.simplefilter("ignore")
        if (scaling > 1):
            return rescale(img, scaling, order=order,
                           anti_aliasing=anti_aliasing,
                           preserve_range=True).astype(np.float32)

        elif (scaling < 1) and integer_inverse:
            return block_resize(img, scaling).astype(np.float32)

        elif (scaling < 1) and not integer_inverse:
            return rescale(img, scaling, anti_aliasing=False,
                           preserve_range=True).astype(np.float32)

        else:  # scaling == 1
            return img


def block_resize(img, scaling):
    """Downsample array, by a factor 'scaling' using a block average that
    ignores nan values. Supported scaling values are scaling values that
    are inverse of integers, plus 0.6"""
    if scaling >= 1:
        raise ValueError("'scaling' value should be less than 1"
                         " for block resize")
    if (1 / scaling) % 1 != 0:
        raise ValueError('Scaling factor: {} not supported.'
                         .format(scaling))

    block_size = round(1/scaling), round(1/scaling)
    return skimage.measure.block_reduce(img, block_size, np.nanmean)
