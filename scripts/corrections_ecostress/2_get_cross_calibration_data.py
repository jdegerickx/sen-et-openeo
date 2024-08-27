"""
    This script downloads the identified ECOSTRESS data for cross-calibration
    and estimates the correction parameters for the S3 LST data 
    through comparison with ECOSTRESS data.

    Instructions for use:

    1. Create a configuration file in the config folder called "configure_cross_cal_data.json"
    {"wasdi_initialisation_file" : specify the path of the wasdi initialisation file (e.g.  "/home/louis_snyders/PycharmProjects/Aries_Ecostress/config/wasdi_configuration_file.json".)
    "workspaces": specify list with names of wasdi workspaces from which you want to import data 
    (e.g. ["Ecostress simultaneous with S3 for 35JLM",
    "Ecostress simultaneous with S3 for 35JMM",
                "Ecostress simultaneous with S3 for 35JNM"]

    "validation_folder" : Specify the folder where to place the resulting GEOTIFF files (e.g. "/vitodata/aries/validation_data/niger") ,
    "validation_satellite": Specify the satellite (e.g. "Ecostress")
    "senet_folder" : specify the location of the sen-et output folder. (e.g. "/vitodata/aries/niger/")

    "include_SESA" : set to true/false to include/ not include in csv file,
    "include_VZA_difference": set to true/false to include/ not include in csv file,
    "generate_csv": set to true/false to generate/not generate csv file,
    "generate_all_data_csv": set to true/false to generate overall csv file for all matches in the folders (recommended),
    "generate_difference_geotiff": set to true/false to generate difference geotiffs (not recommended),
    "skip_processed" : set to true/false,

    "number_of_csv_samples": specify the number of all data samples to be generated.

    }

2. Create a wasdi configuration file in the config folder called "wasdi_configuration_file"
    specify user credentials and the name of the Wasdi workspace that contains the necessary files.
    (e.g.{  "USER": place your wasdi e-mail login (e.g. "louis.snyders@vito.be")
            "PASSWORD": place your wasdi password here (e.g. "PASSWORD123")}
            "WORKSPACE": choose one (e.g. "Ecostress simultaneous with S3 for 31PDR")
            })

    """


import os
import pandas as pd

from sen_et_openeo.lst_cor.utils import (read_configuration_file,
                                         get_ecostress_files_wasdi,
                                         upscale_s3hr_folder_for_validation_data,
                                         comparison)
from sen_et_openeo.lst_cor import (cross_calibration,
                                   directionality)
from sen_et_openeo.lst_cor import csv_analysis


def get_correction_parameters(folder):
    sample_csvs = [file for file in os.listdir(
        folder) if "all_data_sample" in file]

    for sample in sample_csvs:
        df = pd.read_csv(os.path.join(folder, sample),
                         usecols=csv_analysis.COLUMNS)
        gain, offset, r_squared = cross_calibration.s3_lst_bias_df(
            df, conf_int_low=0.1, conf_int_up=0.9)
        dir_par, dir_par_std = directionality.get_baseshape_par(
            df, gain=gain, offset=offset, dir_model=directionality.vinnikov_model)

        # Create a text file for each sample with the parameters
        with open(os.path.join(folder, f"{sample}_parameters.txt"), "w") as param_file:
            param_file.write(f"Gain: {gain[0]}\n")
            param_file.write(f"Offset: {offset}\n")
            param_file.write(
                f"R² of cross_calibration (gain & offset): {r_squared}\n")
            param_file.write(f"Directionality Parameters: {dir_par[0]}\n")
            param_file.write(
                f"Directionality Parameters standard deviation: {dir_par_std[0][0]}\n")


def main(config):

    # Download Geotifs from WASDI to validation folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wasdi_config_path = os.path.join(
        script_dir, "..", "config", 'wasdi_configuration_file.json')
    get_ecostress_files_wasdi(config, wasdi_config_path)

    # Resample HR S3 LST to validation LST and save
    upscale_s3hr_folder_for_validation_data(config)

    # Get difference of simultaneous observations
    comparison(config)

    validation_folder = os.path.join(config.get(
        "validation_folder"), config.get("validation_satellite"))

    # Get sample csv
    csv_analysis.create_sample_csvs_from_validation_folder(validation_folder,
                                                           number_of_samples=config.get(
                                                               "number_of_csv_samples"))

    # Calculate Gain, Offset and Directional parameter
    get_correction_parameters(validation_folder)

    return


if __name__ == '__main__':

    config = read_configuration_file("configure_cross_cal_data")
    main(config)
