import xarray as xr
import numpy as np
from openeo.udf import XarrayDataCube

CONFIDENCE_IN_BAND_DEFAULT = 'confidence_in'
BANDS_TO_MASK_DEFAULT = ['LST']


def get_confidence_in_mask(input_array: np.ndarray, text_code: str) -> np.ndarray:
    """
    Retrieves confidence values from an input array based on the specified text code.

    Args:
        input_array (np.ndarray): Input NumPy array containing confidence values.
        text_code (str): Text code specifying the type of confidence to retrieve. Possible values are:
            'coastline', 'ocean', 'tidal', 'land', 'inland_water', 'unfilled', 'spare',
            'cosmetic', 'duplicate', 'day', 'twilight', 'sun_glint', 'snow',
            'summary_cloud', and 'summary_pointing'.

    Returns:
        np.ndarray: A boolean NumPy array representing confidence values based on the specified text code.
    """
    text_code_to_bit = {'coastline': 0,
                        'ocean': 1,
                        'tidal': 2,
                        'land': 3,
                        'inland_water': 4,
                        'unfilled': 5,
                        'spare': 7,
                        'cosmetic': 8,
                        'duplicate': 9,
                        'day': 10,
                        'twilight': 11,
                        'sun_glint': 12,
                        'snow': 13,
                        'summary_cloud': 14,
                        'summary_pointing': 15
                        }

    if text_code not in text_code_to_bit.keys():
        raise ValueError(f'Could not find {text_code} in "text_code_to_bit"')

    shifts = text_code_to_bit[text_code]

    shifted_data = np.right_shift(input_array, shifts)
    return (shifted_data % 2).astype(bool)


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    Masks clouds in an S3 datacube based on confidence_in.

    Args:
        cube (XarrayDataCube): The input datacube to be processed.
        context (dict): A dictionary containing configuration parameters.
            - 'confidence_in_band': The band used to determine cloud confidence. Defaults to CONFIDENCE_IN_BAND_DEFAULT.
            - 'bands_to_mask': The list of bands to apply the cloud mask to. Defaults to BANDS_TO_MASK_DEFAULT.
    """

    xar_in = cube.get_array()

    if 'confidence_in_band' in context.keys():
        confidence_in_band = context['confidence_in_band']
    else:
        confidence_in_band = CONFIDENCE_IN_BAND_DEFAULT

    if 'bands_to_mask' in context.keys():
        bands_to_mask = context['bands_to_mask']
    else:
        bands_to_mask = BANDS_TO_MASK_DEFAULT

    bands = xar_in['bands'].values.tolist()
    for band in bands_to_mask:
        if band not in bands_to_mask:
            raise KeyError(f'band {band} not found in {bands}')

    output_data_list = list()

    for t in xar_in['t'].values:
        # Get cloud mask
        confidence_in = xar_in.sel(t=t, bands=confidence_in_band)
        confidence_in_uint16 = confidence_in.values.astype(np.uint16)
        cloud_data = get_confidence_in_mask(
            confidence_in_uint16, 'summary_cloud')

        # Mask clouds, recreate original structure for this t
        t_out = list()
        for band in xar_in['bands'].values:
            band_data = xar_in.sel(t=t, bands=band)
            if band in bands_to_mask:
                band_data = band_data.where(np.logical_not(cloud_data))

            t_out.append(band_data)
        output_data_list.append(xr.concat(t_out, dim='bands'))

    xar_out = xr.concat(output_data_list, dim='t')
    return XarrayDataCube(xar_out)
