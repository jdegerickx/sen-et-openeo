import numpy as np
from openeo.udf import XarrayDataCube
from datetime import datetime

CONFIDENCE_IN_BAND_DEFAULT = 'confidence_in'
MAX_CLOUD_PERCENTAGE_DEFAULT = 95.0


def get_confidence_in_mask(input_array: np.ndarray,
                           text_code: str) -> np.ndarray:
    """
    Retrieves confidence values from an input array
    based on the specified text code.

    Args:
        input_array (np.ndarray): Input NumPy array containing
        confidence values.
        text_code (str): Text code specifying the type of confidence
        to retrieve. Possible values are:
            'coastline', 'ocean', 'tidal', 'land', 'inland_water', 'unfilled',
            'spare','cosmetic', 'duplicate', 'day', 'twilight', 'sun_glint',
            'snow','summary_cloud', and 'summary_pointing'.

    Returns:
        np.ndarray: A boolean NumPy array representing confidence values based
        on the specified text code.
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
    return (shifted_data % 2).astype(np.bool_)


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    Filters an XarrayDataCube to retain the least cloudy observation per day.

    Args:
        cube (XarrayDataCube): The input data cube to be processed.
        context (dict): context dictionary:
            - 'confidence_in_band' (str): Optional. The band to use for cloud
               confidence. If not provided, a default value is used.


    """

    xar_in = cube.get_array()

    if 'confidence_in_band' in context.keys():
        confidence_in_band = context['confidence_in_band']
    else:
        confidence_in_band = CONFIDENCE_IN_BAND_DEFAULT

    if 'max_cloud_percentage' in context.keys():
        max_cloud_percentage = context['max_cloud_percentage']
    else:
        max_cloud_percentage = MAX_CLOUD_PERCENTAGE_DEFAULT

    # Dict with date object as key, and tuple of t and percentage as values
    least_clouds = dict()

    # Step1: Fill the least clouds dictionary
    for t in xar_in['t'].values:
        # Get cloud mask
        confidence_in = xar_in.sel(t=t, bands=confidence_in_band)
        confidence_in_uint16 = confidence_in.values.astype(np.uint16)
        cloud_data = get_confidence_in_mask(
            confidence_in_uint16, 'summary_cloud')

        # Get percentage of clouded pixels
        cloud_percentage = np.mean(cloud_data) * 100

        # Only one observation per day is allowed
        dt = datetime.utcfromtimestamp(t.astype('datetime64[s]').astype(int))
        d = dt.date()
        if d not in least_clouds.keys():
            least_clouds[d] = (t, cloud_percentage)
        else:
            _, prev_cloud_percentage = least_clouds[d]
            if cloud_percentage < prev_cloud_percentage:
                least_clouds[d] = (t, cloud_percentage)

    # Step2: remove data that is not in the t_selected list

    # Convert to list
    t_selected = list()
    for t, cloud_percentage in least_clouds.values():
        if cloud_percentage <= max_cloud_percentage:
            t_selected.append(t)

    xar_out = xar_in.sel(t=t_selected)

    return XarrayDataCube(xar_out)
