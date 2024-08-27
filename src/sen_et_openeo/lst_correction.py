import numpy as np
import rasterio
from rasterio.enums import Resampling


def apply_dir_corr(lst_array, vza_array, vinnikov_parameter):
    def vinnikov_function(vza):
        vza = np.radians(vza)
        return 1 - np.cos(vza)

    # Find indices of non-zero values in lst_array
    non_zero_indices = np.nonzero(lst_array)
    # Create a copy of lst_array to store the result
    dir_cor_lst = np.copy(lst_array)
    # Calculate directional effects only for non-zero values
    dir_effects = vinnikov_parameter * vinnikov_function(vza_array)

    # Apply directional correction formula only to non-zero values
    dir_cor_lst[non_zero_indices] -= dir_effects[non_zero_indices]

    return dir_cor_lst


def apply_cross_calibration(s3_lst, gain, offset):
    # Find indices of non-zero values
    non_zero_indices = np.nonzero(s3_lst)

    # Create a copy of the input array to store the result
    result = np.copy(s3_lst)

    # Apply cross calibration formula only to non-zero values
    result[non_zero_indices] = (result[non_zero_indices] - offset) / gain

    return result


def _lst_correction(lst_file, vza_file, output_file, cross_cal_gain=None,
                    cross_cal_offset=None, vinnikov_parameter=None):

    # Read lst
    with rasterio.open(lst_file, 'r') as src:
        lst = src.read(1)
        outprofile = src.profile.copy()
        scale = src.scales[0]
        offset = src.offsets[0]

    # apply scaling & nodata value
    lst = lst * scale + offset
    lst[lst == outprofile['nodata']] = np.nan

    # Read VZA
    with rasterio.open(vza_file, 'r') as src:
        vza = src.read(1)
        profile = src.profile.copy()
        scale_vza = src.scales[0]
        offset_vza = src.offsets[0]

    # apply scaling & nodata value
    vza = vza * scale_vza + offset_vza
    vza[vza == profile['nodata']] = np.nan

    # Apply cross calibration
    if (cross_cal_gain is not None) and (cross_cal_offset is not None):
        lst = apply_cross_calibration(
            lst, cross_cal_gain, cross_cal_offset)

    # Apply directional correction
    if vinnikov_parameter is not None:
        lst = apply_dir_corr(lst, vza, vinnikov_parameter)

    # Apply scaling.
    lst = lst / scale
    lst = np.nan_to_num(lst, outprofile['nodata'])
    lst = lst.astype(np.dtype(outprofile['dtype']))

    # Save corrected LST
    with rasterio.open(output_file, 'w', **outprofile) as output_handler:
        output_handler.scales = [scale]
        output_handler.offsets = [offset]
        output_handler.descriptions = ['LST_CORRECTED']
        output_handler.units = ['K']
        output_handler.write(lst, 1)
        # Generate overviews
        output_handler.build_overviews(
            (4, 8, 16), Resampling.average)

    return
