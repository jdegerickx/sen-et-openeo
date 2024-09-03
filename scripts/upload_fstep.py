# INSTRUCTIONS FOR UPLOADING DATA TO FOOD SECURITY EXPLORER PLATFORM

# STEP 1: Prepare your GeoTIFFs to be uploaded
# (this step is taken care of by sen-et-openeo workflow)
# o	For better performance, first, you need to transform all your TIF files
#   to COG (cloud-optimized GeoTIFF).
# o	Ensure that all GeoTIFF files have a CRS (coordinate reference system).
# o	Place all GeoTIFF files in their own dedicated directory (only these files
#   and nothing extra).

# STEP 2: Fill the CSV table
# (this step is taken care of by sen-et-openeo workflow)
# o	Once you have all files prepared and in a dedicated directory/folder,
#   take the csv-example.csv table (attached) and fill it.
#   Since you are uploading only GeoTIFFs, the geometry column does not
#   need to be filled.
#   For every file, you need the fields (columns) startTime, endTime,
#   filename, and description.
# o	IMPORTANT: save the file with ; as field separator and  not ,

# STEP 3: Create a collection on FOOD SECURITY EXPLORER PLATFORM
# o	Create a collection in the FSX-DAMA UI
#   (https://foodsecurity-tep.ope.insula.earth/dama).
#   To do this, you have to log in,
#   then go to DAMA/Data Discovery/My Storage/Uploaded Data/.
# o	Click on the button "New folder"
#   (it has an icon of a folder with a plus sign + inside).
# o	Add a name and description. For data type, use generic.

# STEP 4: Prepare your access credentials and paths
# o	Get the env_example.txt file (located in the same folder as this script)
#   and rename it to .env
# o	Add your FSX-API credentials in this file.
#   The variables you need to change/add are USERNAME, PASSWORD,
#   COLLECTION_NAME (defined in step 3), PATH_TO_CSV (see step 2),
#   FILES_INPUT_DIR (see step 1).

# STEP 5: Prepare your Python environment
#   (this step is taken care of if you have set up a python environment
#   using the requirements.txt located within this repository)
# o	Create a new Python virtual environment.
# o	Install all needed packages defined in the attached file requirements.txt
#   (e.g., $ pip install -r requirements.txt).

# STEP 6: Execute the script
# o	Monitor the upload. The script will output the files that were uploaded.
# o	If debugging is needed, use the automatically generated log file
#   upload-collection-files.log for more information.
#   This file is generated in the same directory where the Python script
#   was executed.

# NOTE: In case the upload procedure is interrupted, remove the already
# uploaded files from the CSV file and start the script again.

# STEP 7: Check data on FSX-DAMA
# o	Once the upload has finished successfully, you can go back to the web UI
#   and check your data. All geo-files should be georeferenced and can be
#   searched by time, name, and AOI.

# STEP 8: Share GeoTIFF styles
# o	If you want to have the GeoTIFF files with specific styles,
#   please export the style in the SLD (Styled Layer Descriptor) using QGIS.
# o	Share the SLD file with Food Security Explorer administrator.


import os
import requests
from lxml import html
from urllib.parse import urlparse
from urllib.parse import parse_qs
import logging
from logging import handlers
from dotenv import load_dotenv
from requests_toolbelt.multipart.encoder import MultipartEncoder
import pandas as pd
import json


# load env variables
load_dotenv("./.env")

# set variables from loaded env variables
fsx_api_host = os.getenv("FSX_API_HOST")
realm = os.getenv("REALM")
keycloak_jwks_url = os.getenv("KEYCLOAK_JWKS_URL")
keycloak_url = os.getenv("KEYCLOAK_URL")
redirect_url = os.getenv("REDIRECT_URL")
clientid = os.getenv("CLIENTID")
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
collection_name = os.getenv("COLLECTION_NAME")
file_type = os.getenv("FILE_TYPE")
path_to_csv = os.getenv("PATH_TO_CSV")
files_input_dir = os.getenv("FILES_INPUT_DIR")

# set request url variable
upload_endpoint = f"{fsx_api_host}/platformFiles/refData"

# define logger
logger = logging.getLogger("upload-collection-files")
logger.setLevel(logging.DEBUG)
rfh = handlers.RotatingFileHandler(
    "./upload-collection-files.log",
    maxBytes=1e6,
    backupCount=10,
)
rfh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
rfh.setFormatter(formatter)
logger.addHandler(rfh)


def get_auth_token() -> str:
    # obtain access token from a public client
    get_params = {
        "client_id": clientid,
        "redirect_uri": redirect_url,
        "scope": "openid",
        "response_type": "code",
    }

    # create session
    session = requests.Session()

    # session is needed for storing cookies.
    # they are essential in the post request bellow
    response = session.get(
        url=f"{keycloak_url}/realms/{realm}/protocol/openid-connect/auth",
        params=get_params,
    )

    # get authentication url string
    auth_url = html.fromstring(response.content.decode()).forms[0].action

    # accesss credentials
    post_data = {"username": username, "password": password}

    response = session.post(
        url=auth_url, data=post_data, allow_redirects=False)

    code = parse_qs(urlparse(response.headers["Location"]).query)["code"][0]

    post_data = {
        "client_id": clientid,
        "redirect_uri": redirect_url,
        "code": code,
        "grant_type": "authorization_code",
    }

    tokens = session.post(
        f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token",
        data=post_data
    ).json()

    access_token = tokens["access_token"]
    token = f"Bearer {access_token}"

    return token


def get_collection_identifier(token: str) -> int:
    # define search endpoint variable
    search_endpoint = f"{fsx_api_host}/collections/search/parametricFind"

    # define headers and fstep api host
    headers = {
        "Authorization": token,
        "Content-Type": "application/hal+json;charset=UTF-8",
    }

    # set request parameters
    params = {"filter": collection_name}

    try:
        resp = requests.get(search_endpoint, headers=headers, params=params)
        resp.raise_for_status()
        res = resp.json()
        collection_identifier = res["_embedded"]["collections"][0][
            "identifier"]
        return collection_identifier

    except Exception as err:
        logger.error(f"collection ID could not be found. {err}")


def post_multipart_request(
    upload_file: str,
    files_directory: str,
    token: str,
    endpoint: str,
    user_properties: dict,
    collection_identifier: str,
    file_type: str,
    content_type: str = None,
):
    files = MultipartEncoder(
        fields={
            "file": (
                upload_file,
                open(f"{files_directory}{upload_file}", "rb"),
                content_type,
            ),
            "fileType": file_type,
            "collection": collection_identifier,
            "userProperties": (None, user_properties, "application/json"),
        }
    )

    headers = {"Authorization": token,
               "Content-Type": files.content_type}

    response = requests.post(url=endpoint, headers=headers, data=files)

    response.raise_for_status()
    print(f"Reference file {upload_file} was uploaded")

    logger.info(f"Reference file {upload_file} was uploaded")


def main():
    # get token
    token = get_auth_token()

    # get collection id
    collection_identifier = get_collection_identifier(token=token)

    # reading the csv file using read_csv
    # storing the data frame in variable called df
    df = pd.read_csv(path_to_csv, delimiter=";")
    mydict = df.to_dict("index")

    try:
        for i in mydict:
            user_properties = {
                # geometry of file in WKT format
                # only for files of type OTHER (no geometry)
                # "geometry": mydict[i]["geometry"],
                # any small string - no more than 25 words
                "description": mydict[i]["description"],
                # startTime must have this format 2020-01-31T00:00:00Z
                "startTime": mydict[i]["startTime"],
                # endTime must have this format 2020-01-31T00:00:00Z
                "endTime": mydict[i]["endTime"],
            }

            json_user_properties = json.dumps(user_properties)

            if file_type == "SHAPEFILE":
                # TODO
                # check wether files are .shp
                post_multipart_request(
                    mydict[i]["filename"],
                    files_input_dir,
                    token,
                    upload_endpoint,
                    json_user_properties,
                    collection_identifier,
                    file_type,
                    "application/zip",
                )

            elif file_type == "GEOTIFF":
                # TODO
                # check wether files are .tif
                post_multipart_request(
                    mydict[i]["filename"],
                    files_input_dir,
                    token,
                    upload_endpoint,
                    json_user_properties,
                    collection_identifier,
                    file_type,
                    "image/tiff",
                )
            elif file_type == "OTHER":
                post_multipart_request(
                    mydict[i]["filename"],
                    files_input_dir,
                    token,
                    upload_endpoint,
                    json_user_properties,
                    collection_identifier,
                    file_type,
                )

            else:
                raise Exception("requested type file does not exist")

    except Exception as err:
        logger.error(f"Platform file could not be uploaded. {err}")


if __name__ == "__main__":
    main()
