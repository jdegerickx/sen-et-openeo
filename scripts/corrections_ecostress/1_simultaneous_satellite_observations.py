"""
This script locates all required ECOSTRESS data to
be used for bias and directionality correction of
sharpened Sentinel-3 LST data. The script then imports
the data into a WASDI workspace for download later on.

Instructions on how to run this script:
1. Create a wasdi account (www.wasdi.net)

2. Create a configuration file in the config folder called "wasdi_configuration_file"
    set the following parameters in a dictionary.
    {"USER": place your wasdi e-mail login (e.g. "louis.snyders@vito.be")
    "PASSWORD": place your wasdi password here (e.g. "PASSWORD123")
    "WORKSPACE": name of the workspace in WASDI to be used}

3. Create a json file in the config folder with name simult_observations.json.
  "sen_et_output_dir" : specify the root output folder of sen-et processing.
  "desired_observations": specify the type of satellite. (e.g. "Ecostress" or "L8")
                (only Ecostress supported at the moment)

4. run this script
    The resulting files will be stored in the f"<<satellite>> simultaneous with S3 for <<tile>>" folder.
    Make sure this workspace doesn't exist before running the code.

"""

import os
from datetime import datetime, timedelta

import wasdi

from sen_et_openeo.lst_cor.utils import (read_configuration_file,
                                         collect_tile_date_boundaries_dict_from_geotiffs)

provider = "AUTO"


def get_satellite_images(bbox, date_time, satellite, bbox_as_list=False):
    if bbox_as_list:
        lat_s = float(bbox[2])
        lon_w = float(bbox[3])
        lat_n = float(bbox[0])
        lon_e = float(bbox[1])

    else:
        asbbox = bbox.split()
        lat_s = float(asbbox[0])
        lon_w = float(asbbox[1])
        lat_n = float(asbbox[3])
        lon_e = float(asbbox[2])
    start_date = date_time.strftime("%Y-%m-%d")
    end_date = start_date

    if satellite.upper() == "ECOSTRESS":
        platform = "ECOSTRESS"
        provider = "AUTO"

    elif satellite.upper() == "L8" or satellite.upper() == "LANDSAT 8" or satellite.upper() == "LANDSAT-8":
        platform = "L8"
        provider = 'ONDA'

    else:
        print("Change satellite name. Name not recognized")
        return

    a_images = wasdi.searchEOImages(platform, start_date, end_date, lat_n, lon_w, lat_s, lon_e,
                                    None, None, None, None, provider)

    return a_images


def check_quasi_simultaneous_satellite_image_and_datetime(image, date_time, satellite, minutes_difference=10):
    date_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    if satellite.upper() == "ECOSTRESS":
        acquisition_time = image['properties']['startDate']
    elif satellite.upper() == "L8":
        acquisition_time = image["properties"]["endposition"]
    else:
        return
    date_time_validation = datetime.strptime(acquisition_time, date_format)
    time_difference_observations = abs(date_time - date_time_validation)

    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(time_difference_observations.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the time information
    formatted_time = f"{hours} hours, {minutes} minutes, and {seconds} seconds"

    if time_difference_observations <= timedelta(minutes=minutes_difference):
        check = True
        print(f"Observation less than 10 minutes from S3: {formatted_time}")
    else:
        check = False
        print(f"Observation more than 10 minutes from S3: {formatted_time}")

    return check


def get_wasdi_data_from_tile_date_bounds(tile_date_bounds_dict, satellite):
    satellite_simult = {}
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the configuration file
    config_file_path = os.path.join(
        script_dir, "..", "config", 'wasdi_configuration_file.json')
    wasdi.init(config_file_path)

    for tile, date_bounds_dict in tile_date_bounds_dict.items():
        simultaneous_images = []
        for date, bounds in date_bounds_dict.items():
            satellite_images = get_satellite_images(
                bounds, date, satellite, bbox_as_list=True)

            for image in satellite_images:
                if check_quasi_simultaneous_satellite_image_and_datetime(image, date, satellite, minutes_difference=10):
                    simultaneous_images.append(image)

        satellite_simult[tile] = simultaneous_images

    return satellite_simult


def import_data_to_wasdi_workspace(eo_images, satellite):
    # Get the list of products in the workspace
    already_in_workspace = wasdi.getProductsByWorkspace(
        wasdi.getActiveWorkspaceId())
    # List of images not yet available
    images_to_import = []

    # For each found image
    for eo_image in eo_images:
        # Get the file Name from the search result
        filename = eo_image["fileName"]
        # If the file name is not yet in the workspace
        if satellite.upper() == "ECOSTRESS":
            desired_file_types = ["EEHTES", "CLOUD", "GEO"]

        elif satellite.upper() == "L8" or satellite.upper() == "LANDSAT 8" or satellite.upper() == "LANDSAT-8":
            desired_file_types = ["L2SP"]
        else:
            return

        if filename not in already_in_workspace and any(file_type in filename for file_type in desired_file_types):
            # Add it to the list of images to import
            images_to_import.append(eo_image)

    # If there are images to import
    if len(images_to_import) > 0:
        # Trigger the import of the images
        wasdi.importProductList(images_to_import, provider)
        wasdi.wasdiLog("Images Imported")

    return


def create_workspace_if_not_exists(workspace_name):
    """
     Create a Wasdi workspace if it does not exist, and set it as the active workspace.

     Args:
         workspace_name (str): Name of the workspace to be created or set as active.

     Returns:
         None
     """
    current_workspaces = [workspace["workspaceName"]
                          for workspace in wasdi.getWorkspaces()]
    if workspace_name in current_workspaces:
        wasdi.setActiveWorkspaceId(workspace_name)
    else:
        wasdi.createWorkspace(workspace_name)
        wasdi.setActiveWorkspaceId(workspace_name)


def create_wasdi_workspace_for_simultaneous_observations(config_file):
    folder_path = config_file.get("sen_et_output_dir")
    tile_date_bounds_dict = collect_tile_date_boundaries_dict_from_geotiffs(
        folder_path)
    satellite = config_file.get("desired_observations")

    sat_data_dict = get_wasdi_data_from_tile_date_bounds(
        tile_date_bounds_dict, satellite)
    for tile, images in sat_data_dict.items():
        workspace_name = f"{satellite} simultaneous with S3 for {tile}"
        create_workspace_if_not_exists(workspace_name)
        import_data_to_wasdi_workspace(images, satellite)

    return


if __name__ == '__main__':
    config = read_configuration_file("simult_observations")
    create_wasdi_workspace_for_simultaneous_observations(config)
