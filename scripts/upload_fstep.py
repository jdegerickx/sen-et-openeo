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

# NOTE: run this script with "fstep" conda environment !!

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
