# sen-et-openeo
Run SenET workflow through OpenEO.

SenET was originally developed within the Sen4ET ESA project (https://www.esa-sen4et.org/).

So far only the LST sharpening part has been included, not the actual computation of ET.

Main script within this repo is "run_lst_ta_tile.py", which generates a thermal water stress indicator
(LST-Ta, defined as land surface temperature - air temperature).

In addition to the LST sharpening as developed within Sen4ET, here we have added additional bias and directionality corrections based on intercomparison of sharpened Sentinel-3 LST data with ECOSTRESS LST data.
All scripts required to compute these correction coefficients can be found in "corrections_ecostress" folder, located in the scripts folder.

At the end of the main script, all required files are prepared to upload the final results to the food security TEP platform. 

Actual data upload can be done using the "upload_fstep.py" script, located in the "scripts" folder.


# Required user accounts
Before being able to execute the main script "run_lst_ta_tile.py", one would need to create the following user accounts:

- Copernicus Data Space Ecosystem --> use the following link to register:
https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=cdse-public&response_type=code&scope=openid&redirect_uri=https%3A//dataspace.copernicus.eu/account/confirmed/1

- Copernicus Climate Data Store:
    -   In case you do not have an ECMWF user account, create one here by clicking the login/register button in the upper right corner: https://cds-beta.climate.copernicus.eu/

    - Now follow the instructions on this page to make sure you save your personalized token on your local machine: https://cds-beta.climate.copernicus.eu/how-to-api 

    - As a last step, you will need to accept the license for the particular product we are downloading here. Visit the following page, scroll down and click the "Accept license" button:
    https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download