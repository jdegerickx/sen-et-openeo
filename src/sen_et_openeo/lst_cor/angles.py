import math
import os.path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def degrees_to_radians(degrees):
    return degrees * (math.pi / 180.0)


def sat1_earth_sat2_angle(vza1, vza2, vaa1, vaa2):
    # convert angles to radians
    vza1_rad = degrees_to_radians(vza1)
    vza2_rad = degrees_to_radians(vza2)
    vaa1_rad = degrees_to_radians(vaa1)
    vaa2_rad = degrees_to_radians(vaa2)

    # calculate cos(angle) using the spherical law of cosines
    cos_sesa = (np.cos(vza1_rad) * np.cos(vza2_rad) +
                np.sin(vza1_rad) * np.sin(vza2_rad) * np.cos(vaa1_rad - vaa2_rad))

    sesa_rad = np.arccos(cos_sesa)

    sesa_deg = sesa_rad * (180.0 / math.pi)

    return sesa_deg


def get_raster_matching_geotiff_values(raster, geotif):
    # Open the raster and geotiff files
    with rasterio.open(raster) as src_raster, rasterio.open(geotif) as src_geotiff:
        # Reproject the geotiff to match the raster's spatial characteristics
        geotiff_reprojected, _ = reproject(
            source=src_geotiff.read(1),
            destination=rasterio.band(src_raster, 1),
            src_transform=src_geotiff.transform,
            src_crs=src_geotiff.crs,
            dst_transform=src_raster.transform,
            dst_crs=src_raster.crs,
            resampling=Resampling.nearest
        )

        # Initialize a new raster with no-fill values
        raster_matching_geotiff = src_raster.read(1, masked=True)

        # Find overlapping indices between the raster and geotiff
        overlap_mask = ~(raster_matching_geotiff.mask |
                         geotiff_reprojected.mask)

        # Assign values where there is overlap
        raster_matching_geotiff[overlap_mask] = geotiff_reprojected[overlap_mask]

        # Assign the no-fill value to areas without overlap
        no_fill_value = src_geotiff.nodatavals[0]
        raster_matching_geotiff[~overlap_mask] = no_fill_value

        return raster_matching_geotiff


def get_valid_values_mask(val_vza, val_vaa, S3_vza, S3_vaa):
    valid_mask = (val_vza <= 180) & (S3_vza <= 180) & (
        val_vaa <= 360) & (S3_vaa <= 360) & (S3_vaa != 0)

    # Check if any of the values is NaN
    nan_mask = ~np.isnan(val_vza) & ~np.isnan(
        S3_vza) & ~np.isnan(val_vaa) & ~np.isnan(S3_vaa)

    resulting_mask = (valid_mask & nan_mask)

    return resulting_mask


def get_sesa_array(val_vza, val_vaa, S3_vza, S3_vaa, valid_mask):
    # Initialize an empty array to store the results
    result_array = np.empty_like(val_vza, dtype=float)

    # Apply the function only to valid and non-NaN values
    result_array[valid_mask] = sat1_earth_sat2_angle(val_vza[valid_mask],
                                                     S3_vza[valid_mask],
                                                     val_vaa[valid_mask],
                                                     S3_vaa[valid_mask])

    # Set invalid or NaN values to NaN in the result
    result_array[~valid_mask] = np.nan

    return result_array


def get_zenith_difference_array(vza1, vza2, valid_mask):
    # Initialize an empty array to store the results
    result_array = np.empty_like(vza1, dtype=float)

    # Apply the function only to valid and non-NaN values
    result_array[valid_mask] = vza1[valid_mask] - vza2[valid_mask]

    # Set invalid or NaN values to NaN in the result
    result_array[~valid_mask] = np.nan

    return result_array


def get_S3_validation_sat_angles(upscaled_S3_file, validation_file, val_vza_index=2, val_vaa_index=3, sesa=True, zenith_dif=True):
    angles = {}

    # Load validation data
    with rasterio.open(validation_file) as validation_dataset:
        val_vza = validation_dataset.read(val_vza_index)
        val_vaa = validation_dataset.read(val_vaa_index)

    # Get S3 angle paths
    basename = os.path.basename(upscaled_S3_file)
    S3_vza_file_path = os.path.join(os.path.dirname(upscaled_S3_file),
                                    basename.replace("upscaled_S3-LSTHR", "original_S3_vza"))
    S3_vaa_file_path = os.path.join(os.path.dirname(upscaled_S3_file),
                                    basename.replace("upscaled_S3-LSTHR", "original_S3_vaa"))

    with rasterio.open(S3_vza_file_path) as S3_vza_file:
        S3_vza = S3_vza_file.read(1)

    with rasterio.open(S3_vaa_file_path) as S3_vaa_file:
        S3_vaa = S3_vaa_file.read(1)

    valid_mask = get_valid_values_mask(val_vza, val_vaa, S3_vza, S3_vaa)

    if sesa:
        sesa_raster = get_sesa_array(
            val_vza, val_vaa, S3_vza, S3_vaa, valid_mask)
        angles["SESA"] = sesa_raster

    if zenith_dif:
        zenith_dif_raster = get_zenith_difference_array(
            S3_vza, val_vza, valid_mask)
        angles["VZA_diff"] = zenith_dif_raster

    return angles


def normalize_angle(angle):
    """
     Normalize an angle to be within the range [0, 360) degrees."""
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle
