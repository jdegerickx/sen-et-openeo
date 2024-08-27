# sen-et-openeo
Run SenET workflow on openeo.

SenET was originally developed within the Sen4ET ESA project (https://www.esa-sen4et.org/).

So far only the LST sharpening part has been included, not the actual computation of ET.

Main script within this repo is "run_lst_ta_tile.py", which generates a thermal water stress indicator
(LST-Ta, defined as land surface temperature - air temperature).

In addition to the LST sharpening as developed within Sen4ET, here we have added additional bias and directionality corrections based on intercomparison of sharpened Sentinel-3 LST data with ECOSTRESS LST data.
All scripts required to compute these correction coefficients can be found in "corrections_ecostress" folder, located in the scripts folder.

At the end of the main script, all required files are prepared to upload the final results to the food security TEP platform. 

Actual data upload can be done using the "upload_fstep.py" script, located in the "scripts" folder.