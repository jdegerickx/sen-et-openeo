import json
import os
from pathlib import Path
from datetime import datetime
import rasterio
from pyproj import Transformer
import glob
import numpy as np
import pandas as pd
from osgeo import gdal

import wasdi

from eotile.eotile_module import quick_search as tile_search

from sen_et_openeo.lst_cor.georeference_ecostress import georeference_ecostress
from sen_et_openeo.lst_cor.resampling import (WarpS3HRtoValidation,
                                              get_acquistion_angles_files_S3)
from sen_et_openeo.lst_cor.time import (get_datetime_from_yyyymmddthhmmss,
                                        find_simult_acquisitions,
                                        extract_datetime_strings_from_filename)
from sen_et_openeo.lst_cor.angles import (normalize_angle,
                                          sat1_earth_sat2_angle)
from sen_et_openeo.lst_cor.validation_tools import get_difference_geotiff
from sen_et_openeo.lst_cor.world_cover import get_matching_world_cover_raster


# TODO: subset to S3 window to increase speed.
def generate_validation_csv(s3hr_filepath, validation_image_path, csv_path, tile=None):
    def change_pov_validation_azimuths():
        saa_difference = abs(
            np.mean(df["original_S3_saa"]) - np.mean(df["validation_solar_azimuth"]))
        if saa_difference < 10:
            pass
        elif 165 < saa_difference < 195:
            df["validation_solar_azimuth"] = df["validation_solar_azimuth"] - 180
            df["validation_view_azimuth"] = df["validation_view_azimuth"] - 180

        else:
            raise Exception(
                f"Something is wrong with the data. The solar azimuth angles of the S3 file {s3hr_filepath} doesn't correspond to the"
                f" validation file {validation_image_path}.")

    def get_flat_array_from_file(filepath, band_idx=None, valid_indices_list=None, tile_bounds=None):
        dataset = gdal.Open(filepath)

        if band_idx:
            raster_name = dataset.GetRasterBand(band_idx).GetDescription()

        else:
            band_idx = 1
            raster_name = None

        raster_band = dataset.GetRasterBand(band_idx)
        #
        # # Open the mask GeoTIFF file if provided
        # if tile_bounds:
        #     xmin, ymin, xmax, ymax = tile_bounds
        #     transform = dataset.GetGeoTransform()
        #     pixel_xmin = int((xmin-transform[0])/transform[1])
        #     pixel_ymax = int((ymin-transform[3])/transform[5])
        #     pixel_xmax = int((xmax-transform[0])/transform[1])
        #     pixel_ymin = int((ymax-transform[3])/transform[5])
        #
        #     raster_array = np.array(raster_band.ReadAsArray(pixel_xmin, pixel_ymin,
        #                                                     pixel_xmax-pixel_xmin, pixel_ymax-pixel_ymin))
        #
        # else:
        raster_array = np.array(raster_band.ReadAsArray())

        if raster_band.GetScale():
            raster_array = np.float32(raster_array) * \
                np.float32(raster_band.GetScale())

        flattened_array = raster_array.flatten()

        if valid_indices_list:
            flattened_array = flattened_array[valid_indices_list]

        dataset = None
        return flattened_array, raster_name

    val_band_name_change = {"validation_view_zenith": "validation_vza",
                            "validation_view_azimuth": "validation_vaa",
                            "validation_solar_zenith": "validation_sza",
                            "validation_solar_azimuth": "validation_saa"}

    angles_S3 = ["original_S3_vza",
                 "original_S3_vaa",
                 "original_S3_sza",
                 "original_S3_saa"]

    s3_time, ecostress_time = extract_datetime_strings_from_filename(
        s3hr_filepath)
    dir_path = os.path.dirname(s3hr_filepath)
    basename_s3hr = os.path.basename(s3hr_filepath)
    angles_S3_filepaths = {angle: os.path.join(dir_path, basename_s3hr.replace("upscaled_S3-LSTHR", angle)) for angle in
                           angles_S3}
    # if tile:
    #     tile_bounds = get_bounds_from_s2tile(tile)
    # else:
    #     tile_bounds = None

    lst_S3hr, _ = get_flat_array_from_file(s3hr_filepath)
    lst_val, _ = get_flat_array_from_file(validation_image_path, band_idx=1)

    valid_indices = np.where((lst_S3hr != 0) & (lst_val != 0))
    # Create a Pandas DataFrame
    df = pd.DataFrame()
    # If there is no valid index, an empty csv is created and the rest of the function is skipped.
    if len(valid_indices[0]) == 0:
        print(f"No match for {basename_s3hr}")
        df.to_csv(csv_path, index=False)
        return False

    lst_S3hr = lst_S3hr[valid_indices]
    df["lst_S3hr"] = lst_S3hr
    del lst_S3hr

    for i in range(len(angles_S3)):
        flat_angles, _ = get_flat_array_from_file(
            angles_S3_filepaths[angles_S3[i]], valid_indices_list=valid_indices)
        df[angles_S3[i]] = flat_angles
        del flat_angles

    for i in range(1, 8):
        flat_array, name = get_flat_array_from_file(
            validation_image_path, band_idx=i, valid_indices_list=valid_indices)
        df[f"validation_{name}"] = flat_array
        del flat_array

    # Check the orientation of the azimuth angles
    if np.isnan(np.nanmean(df["original_S3_saa"])) is False:
        change_pov_validation_azimuths()

    land_cover = get_matching_world_cover_raster(s3hr_filepath)
    flat_land_cover = land_cover.flatten()
    land_cover = flat_land_cover[valid_indices]
    df["land_cover"] = land_cover

    df["SESA"] = df.apply(lambda row: sat1_earth_sat2_angle(row['original_S3_vza'],
                                                            row['validation_view_zenith'],
                                                            row['original_S3_vaa'],
                                                            row['validation_view_azimuth']), axis=1)
    df["S3_time"] = s3_time
    df["validation_time"] = ecostress_time

    for azimuth_angles in ["original_S3_vaa", "original_S3_saa", 'validation_view_azimuth',  'validation_solar_azimuth']:
        df[azimuth_angles] = df[azimuth_angles].apply(normalize_angle)

    df.rename(columns=val_band_name_change, inplace=True)

    df.to_csv(csv_path, index=False)
    df = None
    return True


def create_difference_images(config):
    folder = config.get("folder")
    tiles = os.listdir(folder)
    for tile in tiles:
        upscaled_s3_tile_path = os.path.join(
            folder, tile, "upscaled_S3-LSTHR_to_Ecostress")
        validation_tile_path = os.path.join(folder, tile, "LST")

        subfolders = os.listdir(upscaled_s3_tile_path)

        for observation_folder in subfolders:
            observation_folder_path = os.path.join(
                upscaled_s3_tile_path, observation_folder)
            files = os.listdir(observation_folder_path)

            for file in files:
                s3hr = file
                s3hr_no_ext, ext = os.path.splitext(s3hr)
                s3hr_filepath = os.path.join(observation_folder_path, s3hr)

                # s3hr_datetime = get_datetime_from_yyyymmddthhmmss(observation_folder)
                # validation_simult = find_simult_acquisitions(s3hr_datetime,validation_tile_path)
                # validation_simult = [validation_simult[1]]
                validation_image = s3hr_no_ext[-15:]

                # for validation_image in validation_simult:
                validation_image_path = os.path.join(
                    validation_tile_path, validation_image, f"{validation_image}.tif")

                difference_filename = f"{s3hr_no_ext[:39]}_minus_validation_{validation_image}.tif"
                outdir = os.path.join(
                    folder, tile, "S3_validation_difference", s3hr_no_ext[24:39])

                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                file_path = os.path.join(outdir, difference_filename)
                get_difference_geotiff(
                    s3hr_filepath, validation_image_path, file_path)

    return


def combine_csv_files(filepaths, output_filepath, chunk_size=100000):
    combined_dataframe = None
    chunk_count = 0

    for filepath in filepaths:
        # Check if the file has content
        if os.path.getsize(filepath) > 1:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)

            # Concatenate the DataFrame to the combined DataFrame
            if combined_dataframe is None:
                combined_dataframe = df
            else:
                combined_dataframe = pd.concat(
                    [combined_dataframe, df], ignore_index=True)

            # Check if the combined DataFrame size exceeds the chunk size
            if combined_dataframe.shape[0] >= chunk_size:
                # Save the combined DataFrame to the CSV file
                if chunk_count == 0:
                    combined_dataframe.to_csv(output_filepath, index=False)
                else:
                    combined_dataframe.to_csv(
                        output_filepath, mode='a', index=False, header=False)
                chunk_count += 1

                # Reset the combined DataFrame
                combined_dataframe = None

    # Save any remaining data
    if combined_dataframe is not None:
        if chunk_count == 0:
            combined_dataframe.to_csv(output_filepath, index=False)
        else:
            combined_dataframe.to_csv(
                output_filepath, mode='a', index=False, header=False)

    print("Combined CSV file saved to:", output_filepath)


def comparison(config):
    # folder = config.get("folder")
    folder = os.path.join(config.get("validation_folder"),
                          config.get("validation_satellite"))
    tiles = os.listdir(folder)

    for tile in tiles:
        print(f"{tile} is being handled.")
        csv_filepaths = []

        upscaled_s3_tile_path = os.path.join(
            folder, tile, "upscaled_S3-LSTHR_to_Ecostress")
        S3_subfolders = os.listdir(upscaled_s3_tile_path)

        validation_tile_path = os.path.join(folder, tile, "LST")

        for observation_folder in S3_subfolders:
            observation_folder_path = os.path.join(
                upscaled_s3_tile_path, observation_folder)
            S3_files = os.listdir(observation_folder_path)
            S3_files = [file for file in S3_files if "LST" in file and os.path.splitext(file)[
                1] == ".tif"]
            for s3hr in S3_files:
                s3hr_no_ext, ext = os.path.splitext(s3hr)
                s3hr_filepath = os.path.join(observation_folder_path, s3hr)

                validation_image = s3hr_no_ext[-15:]
                validation_image_path = os.path.join(
                    validation_tile_path, validation_image, f"{validation_image}.tif")

                outdir = os.path.join(
                    folder, tile, "S3_validation_difference", s3hr_no_ext[24:39])
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                csv_filepath = os.path.join(outdir, f"{s3hr_no_ext}.csv")

                if config.get("skip_processed"):
                    csv_filepaths.append(csv_filepath)
                    if os.path.exists(csv_filepath):
                        print(f"{csv_filepath} already exists.")
                        pass

                    elif config.get("generate_csv"):
                        print(f"Begin creation of {csv_filepath}")
                        generate_validation_csv(
                            s3hr_filepath, validation_image_path, csv_filepath, tile=tile)
                        print(f"{csv_filepath} created")
                else:
                    print(f"Begin creation of {csv_filepath}")
                    generate_validation_csv(
                        s3hr_filepath, validation_image_path, csv_filepath, tile=tile)
                    print(f"{csv_filepath} created")

                if config.get("generate_difference_geotiff"):
                    difference_filename = f"{s3hr_no_ext[:39]}_minus_validation_{validation_image}.tif"
                    file_path = os.path.join(outdir, difference_filename)
                    get_difference_geotiff(
                        s3hr_filepath, validation_image_path, file_path)

        if config.get("generate_all_data_csv"):
            all_data_path = os.path.join(folder, tile, "all_data.csv")

            if config.get("skip_processed") and os.path.exists(all_data_path):
                print(f"{all_data_path} already exists. It is not overwritten")
                pass
            else:
                combine_csv_files(csv_filepaths, all_data_path)

    return


def get_val_tile_geotiff_dictionary(validation_satellite_path):
    tiles = os.listdir(validation_satellite_path)

    tile_geotiff_dict = {}
    for tile in tiles:
        validation_lst_folder = os.path.join(
            validation_satellite_path, tile, "LST")
        geotiff_folders = os.listdir(validation_lst_folder)
        geotiff_paths = [os.path.join(
            validation_lst_folder, geotiff_folder) for geotiff_folder in geotiff_folders]
        tile_geotiff_dict[tile] = geotiff_paths

    return tile_geotiff_dict


def get_s3_simultaneous_with_validation(validation_list, s3hr_folder):
    path_pairs = {}

    for validation_path in validation_list:
        full_path_no_ext, ext = os.path.splitext(validation_path)
        path, file_name = os.path.split(full_path_no_ext)
        validation_time = get_datetime_from_yyyymmddthhmmss(file_name)
        s3hr_simult = find_simult_acquisitions(validation_time, s3hr_folder)
        s3hr_simult_path = [os.path.join(
            s3hr_folder, subfolder) for subfolder in s3hr_simult]
        path_pairs[validation_path] = s3hr_simult_path

    return path_pairs


def upscale_s3_for_sat_im_pairs(sat_im_pairs, out_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for val_path in sat_im_pairs:
        for s3hr_path in sat_im_pairs[val_path]:
            val_file_path = os.path.join(val_path, os.listdir(val_path)[0])
            filename = os.listdir(s3hr_path)[0]
            s3hr_file_path = os.path.join(s3hr_path, filename)
            filename = filename.replace("S3", "upscaled_S3")
            filename = filename.replace(
                ".tif", f"_to_ecostress_{os.listdir(val_path)[0]}")
            subfoldername = os.path.basename(s3hr_path)
            subfolderpath = os.path.join(out_dir, subfoldername)
            if not os.path.exists(subfolderpath):
                os.makedirs(subfolderpath)
            out_path = os.path.join(subfolderpath, filename)

            if os.path.exists(out_path):
                pass
            else:
                angle_files = get_acquistion_angles_files_S3(out_path, "/vitodata/aries/data/S3_LST/",
                                                             include_cloud_MSK=False)
                WarpS3HRtoValidation(
                    s3hr_file_path, out_path, val_file_path, additional_bands=angle_files)

    return


def upscale_s3hr_folder_for_validation_data(configure):
    senet_path, validation_path = configure.get(
        "senet_folder"), configure.get("validation_folder")
    validation_satellite = configure.get("validation_satellite")
    validation_satellite_path = os.path.join(
        validation_path, validation_satellite)

    val_tile_geotiff_paths_dict = get_val_tile_geotiff_dictionary(
        validation_satellite_path)

    for tile, val_geotiff_paths in val_tile_geotiff_paths_dict.items():
        out_dir = os.path.join(validation_path, validation_satellite, tile,
                               f"upscaled_S3-LSTHR_to_{validation_satellite}")

        s3_hr_path_tile = os.path.join(
            senet_path, "cogs_fin", tile, "S3", "S3-LSTHR")

        path_pairs = get_s3_simultaneous_with_validation(
            val_geotiff_paths, s3_hr_path_tile)

        upscale_s3_for_sat_im_pairs(path_pairs, out_dir)

    return


def get_geotiff_bounds(file_path, lat_lon_format=True):
    try:
        with rasterio.open(file_path) as dataset:
            # Get the bounds
            bounds = dataset.bounds
            if lat_lon_format:
                src_crs = dataset.crs
                transformer = Transformer.from_crs(src_crs, "EPSG:4326")
                # bounds = list(bounds)
                bottom, left = transformer.transform(
                    bounds.left, bounds.bottom)
                top, right = transformer.transform(bounds.right, bounds.top)

                bounds = [top, right, bottom, left]

    except:
        print(f"Unable to open {file_path}")
        return None

    return bounds


def collect_s3_filepaths(folder_path):
    # Initialize empty subfolder names dictionary
    file_paths = {}

    # Get list of tiles
    tiles = [tile for tile in os.listdir(folder_path)
             if os.path.isdir(os.path.join(folder_path, tile))]

    # Iterate over tiles
    for tile in tiles:
        indir = Path(folder_path) / tile / '003_sharpening'
        s3_files = sorted(glob.glob(str(indir / '*_fin.tif')))
        if len(s3_files) > 0:
            file_paths[tile] = s3_files
        else:
            file_paths[tile] = None

    return file_paths


def collect_tile_date_boundaries_dict_from_geotiffs(folder_path):
    """
    Collects geotiff boundaries from a specified folder.

    Args:
        folder_path (str): The path to the folder containing geotiff files.

    Returns:
        dict: A dictionary containing geotiff boundaries organized by tile and timestamp.
              The structure of the returned dictionary is {tile: {timestamp: [bounds]}}"""

    bounds_dictionary = {}

    # Collect subfolder names and geotiff paths
    geotiff_paths_dict = collect_s3_filepaths(folder_path)

    # Iterate through tiles and geotiff paths
    for tile, geotiff_paths in geotiff_paths_dict.items():
        tile_dictionary = {}

        # Iterate through geotiff paths for a specific tile
        for geotiff_path in geotiff_paths:
            if geotiff_path is None:
                pass
            else:
                bounds = get_geotiff_bounds(geotiff_path)

                # Extract file information
                filename = os.path.basename(geotiff_path)
                date_time_substr = filename.split('_')[-2]

                file_date = datetime.strptime(
                    date_time_substr, '%Y%m%dT%H%M%S')

                # Store bounds in the tile dictionary with timestamp
                if bounds:
                    tile_dictionary[file_date] = list(bounds)
                else:
                    print(f"{geotiff_path} doesn't work")
                    pass

        # Store the tile dictionary in the overall bounds dictionary
        bounds_dictionary[tile] = tile_dictionary

    return bounds_dictionary


def get_bbox_from_s2_tile(s2_tile):
    """
    :param s2_tile: e.g. "31PDR"
    :return: Bounding box as a string comma separated in form South, West, East, North
    """
    tile = tile_search(s2_tile, "tile_id", "S2")
    bbox = (str(tile.bounds.at[s2_tile, "maxy"]) + " " +
            str(tile.bounds.at[s2_tile, "minx"]) + " " +
            str(tile.bounds.at[s2_tile, "miny"]) + " " +
            str(tile.bounds.at[s2_tile, "maxx"]))
    return bbox


def read_configuration_file(config_name):

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = script_dir.parents[3] / \
        "scripts/sen_et_openeo/corrections_ecostress/config"
    config_path = config_path / f"{config_name}.json"
    with open(config_path, "r") as config_file:
        loaded_config = json.load(config_file)

    return loaded_config


def get_ecostress_files_wasdi(config, wasdi_config_path):

    workspaces = config.get("workspaces")
    geo_corrupted_files = []
    for workspace_name in workspaces:
        tile = workspace_name.split()[-1]
        satellite = workspace_name.split()[0]
        output_dir = config.get("validation_folder")

        wasdi.init(wasdi_config_path)
        wasdi.setActiveWorkspaceId(wasdi.getWorkspaceIdByName(workspace_name))
        wasdi_files = wasdi.getProductsByActiveWorkspace()

        # List directory contents and create lists of ECOSTRESS HDF5 files (GEO, ET)
        geo_list = [file for file in wasdi_files if file.endswith(
            '.h5') and 'GEO' in file]
        eco_list = [file for file in wasdi_files if file.endswith(
            '.h5') and 'LSTE' in file]
        cmask_list = [file for file in wasdi_files if file.endswith(
            '.h5') and 'CLOUD' in file]

        for lst_name in eco_list:
            acquisition_time = lst_name[-26:-11]

            # Check if the file already exists
            filepath = os.path.join(
                output_dir, satellite, tile, "LST", acquisition_time, acquisition_time + ".tif")
            # if not os.path.exists(filepath):
            if not os.path.exists(filepath):
                geo_name = next(
                    (geo_file for geo_file in geo_list if lst_name[-36:-11] in geo_file), None)
                cmask_name = next(
                    (cmask_file for cmask_file in cmask_list if lst_name[-36:-11] in cmask_file), None)

                lst_path, geo_path, cmask_path = wasdi.getPath(
                    lst_name), wasdi.getPath(geo_name), wasdi.getPath(cmask_name)

                filepath = os.path.join(
                    output_dir, satellite, tile, "LST", acquisition_time, acquisition_time + ".tif")

                created = georeference_ecostress(lst_path, geo_path, cmask_path, filepath,
                                                 include_viewing_geometry=True)
                if created == True:
                    print(f"{filepath} created")
                else:
                    print(
                        f"The file {geo_path} is corrupted (L1GEOMetadata/OrbitCorrectionPerformed is False)")
                    geo_corrupted_files.append(geo_path)

                os.remove(lst_path), os.remove(geo_path), os.remove(cmask_path)

            else:
                print(f"{filepath} already exists, skipping")

    for path in geo_corrupted_files:
        print(
            f"The file {path} is corrupted (L1GEOMetadata/OrbitCorrectionPerformed is False)")
