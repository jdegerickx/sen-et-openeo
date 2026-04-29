from pathlib import Path
from typing import Optional, Union, Dict, Callable, List
import gc
import sys
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import re
import pickle
import math
import psutil
from copy import deepcopy
import os

import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
from osgeo import gdal
from rasterio.enums import Resampling
import openeo
from openeo.rest.connection import Connection
from openeo.rest.datacube import DataCube
from openeo.rest.job import BatchJob
from openeo.rest import OpenEoApiError
import loguru
import rioxarray
import xarray as xr
import shutil
import numpy as np
import richdem as rd

from sen_et_openeo.utils.geoloader import OUTPUT_SCALING
from sen_et_openeo.utils.angles import incidence_angle_tilted

# UDF's
import sen_et_openeo.udf.filter_least_clouds_s3
import sen_et_openeo.udf.filter_sza_s3
import sen_et_openeo.udf.mask_clouds_s3

# For sharpening
from sen_et_openeo.utils.timedate import find_closest_date
from sen_et_openeo.utils.pyDMS import DecisionTreeSharpener

# For LST correction
from sen_et_openeo.lst_correction import _lst_correction


class SenETDownload:
    """
        Handles the processing and download of S2, S3 and DEM cubes
        from openEO backend.
        Examples:
            >>> output_path = Path('/data/Users/public/wensd01')
            >>> spatial_extent = = {"west": 5.102634,"east": 5.432847,
                                "south": 51.175899,"north": 51.389673,}
            >>> temporal_extent = ["2020-04-02", "2020-04-20"]
            >>> data_download = SenETDownload(spatial_extent, temporal_extent,
                output_format='gtiff')
            >>> output_files = data_download.download(output_path)
    """
    LOGGER = loguru.logger

    OPENEO_VITO_URL = "openeo.vito.be"
    OPENEO_CDSE_URL = "openeo.dataspace.copernicus.eu"
    # OPENEO_CDSE_URL = "openeo-staging.dataspace.copernicus.eu"
    S2_BAND_NAMES = ["B02", "B03", "B04", "B05",
                     "B06", "B07", "B08", "B8A", "B11", "B12"]
    S3_BAND_NAMES1 = ["LST", "LST_uncertainty", "exception", "confidence_in"
                        ]
    S3_BAND_NAMES2 = [ "sunAzimuthAngles", "sunZenithAngles",
                     "viewAzimuthAngles", "viewZenithAngles"]
    S3_WARP_TO_HR = ["sunAzimuthAngles", "sunZenithAngles",
                     "viewAzimuthAngles", "viewZenithAngles", 'lat', 'lon']
    JOB_OPTIONS_S2 = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "5G",
        "executor-cores": "1",
        "driver-memory": "4G",
        "driver-memoryOverhead": "6G",
    }
    JOB_OPTIONS_S3 = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "1G",
        "executor-cores": "1",
        "python-memory": "2G"
    }
    JOB_OPTIONS_DEM = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "1",
    }

    # ESA WorldCover — version depends on the acquisition year of the data:
    #   before 2021  → ESA_WORLDCOVER_10M_2020_V1  (product year 2020)
    #   2021 onward  → ESA_WORLDCOVER_10M_2021_V2  (product year 2021)
    WORLDCOVER_2020 = 'ESA_WORLDCOVER_10M_2020_V1'
    WORLDCOVER_2021 = 'ESA_WORLDCOVER_10M_2021_V2'
    WORLDCOVER_CUTOFF_YEAR = 2021   # first year that uses the 2021 product
    JOB_OPTIONS_WORLDCOVER = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "1",
    }

    DOWNLOAD_RETRIES = 3  # Retry count when data is not available
    RETRY_DELAY = 30  # In seconds
    BIOPAR_CHUNK_RETRIES = 2  # Extra retries for failed/cancelled chunks

    # UDF's
    UDF_SZA_S3_FILE = sen_et_openeo.udf.filter_sza_s3.__file__
    UDF_FILTER_CLOUD_S3_FILE = sen_et_openeo.udf.filter_least_clouds_s3.__file__
    UDF_MASK_CLOUDS_S3_FILE = sen_et_openeo.udf.mask_clouds_s3.__file__

    # Sharpening
    S2_SHARP_INPUTS = ['B02', 'B03', 'B04',
                       'B05', 'B06', 'B07',
                       'B08', 'B11', 'B12']

    SHARPENING_INPUTS = ['S2-B02', 'S2-B03', 'S2-B04',
                         'S2-B05', 'S2-B06', 'S2-B07',
                         'S2-B08', 'S2-B11', 'S2-B12',
                         'DEM-alt-20m', 'S3-inc']

    # Biophysical variables (biopar) for PyTSEB
    BIOPAR_VARIABLES = ["LAI", "FAPAR", "FCOVER"]
    BIOPAR_UDP_URL = ("https://raw.githubusercontent.com/ESA-APEx/"
                     "apex_algorithms/refs/heads/main/algorithm_catalog/"
                     "vito/biopar/openeo_udp/biopar.json")
    JOB_OPTIONS_BIOPAR = {
        "executor-memory": "7G",
        "executor-memoryOverhead": "4G",
        "executor-cores": "1",
    }

    # TODO: should be configurable in the instance itself.
    NUM_THREADS_GDAL = 'ALL_CPUS'
    NUM_THREADS_RIO = 'all_cpus'
    NUM_JOBS_DISAGGREGATOR = psutil.cpu_count(logical=False)

    def __init__(self,
                 spatial_extent: Union[Dict[str, float], Path, str],
                 temporal_extent: Union[list, tuple],
                 s2_should_mask: bool = True,
                 s2_should_composite: bool = True,
                 s2_should_interpolate: bool = True,
                 s3_should_filter: bool = True,
                 s3_should_mask_clouds: bool = True
                 ):
        """
            Initializes the SenETDownload object with the provided parameters.
            Args:
                spatial_extent (Union[Dict[str, float], Path]): Spatial
                    extent for the datacube query, given as
                    a dictionary with bounds (west, east, north, south)
                    or a Path to a file shapefile (.gpkg). Can also
                    be a valid MGRS tile (e.g. 34HBH)
                temporal_extent (Union[list, tuple]): Temporal extent
                    for the datacube query as a list or tuple of start
                    and end dates.
                s2_should_mask (bool, optional): TODO
                s2_should_composite (bool, optional): TODO
                s2_should_interpolate (bool, optional): TODO
                s3_should_filter (bool, optional): TODO
        """
        self._log = type(self).LOGGER
        self._spatial_extent = type(
            self)._convert_spatial_extent(spatial_extent)
        if isinstance(spatial_extent, str):
            self.tile = spatial_extent
        else:
            self.tile = "Custom"
        self._temporal_extent = temporal_extent
        self._s2_should_mask = s2_should_mask
        self._s2_should_composite = s2_should_composite
        self._s2_should_interpolate = s2_should_interpolate

        self._s3_should_filter = s3_should_filter
        self._s3_should_mask_clouds = s3_should_mask_clouds
        self._s3_pixel_size = (1098.0, -1098.0)  # Optimal = 1098

        self.name_s2 = 'SENTINEL2_L2A'
        self.name_s3 = 'SENTINEL3_SLSTR_L2_LST'
        self.name_dem = 'COPERNICUS_30'

        # Choose WorldCover product version based on temporal extent start year
        start_year = int(str(temporal_extent[0])[:4])
        if start_year < type(self).WORLDCOVER_CUTOFF_YEAR:
            self.name_worldcover = type(self).WORLDCOVER_2020
        else:
            self.name_worldcover = type(self).WORLDCOVER_2021

        self.delete_temp = True

        # Set these values to provide jobid's
        self.s2_jobid: Optional[str] = None
        self.s3_jobid: Optional[str] = None
        self.dem_jobid: Optional[str] = None
        self.worldcover_jobid: Optional[str] = None

        # store download results here from download():
        self._download_results: Optional[dict] = None
        # Store preprocessing result here from preprocess():
        self._preprocess_results: Optional[dict] = None
        # Store sharpening results here from sharpening():
        self._sharpening_results: Optional[dict] = None
        # Store LST correction results here from lst_correction():
        self._lst_results: Optional[dict] = None
        # Store biophysical variable results here from compute_biopar():
        self._biopar_results: Optional[dict] = None

        # See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r
        self.warp_resampling_method: str = 'near'

        # # Quick and dirty method to prompt the user with a single login url
        # openeo.connect(type(self).OPENEO_URL).authenticate_oidc()

    @property
    def _temporal_pkl_suffix(self) -> str:
        """Short string encoding the temporal extent for use in pkl filenames.

        Ensures that runs with different temporal extents (e.g. time chunks)
        produce separate pkl files and never trigger the extent-mismatch error.
        Example: temporal_extent=['2024-05-01','2024-06-30'] → '_20240501_20240630'
        """
        start = str(self._temporal_extent[0]).replace('-', '')[:8]
        end   = str(self._temporal_extent[1]).replace('-', '')[:8]
        return f'_{start}_{end}'

    def run(self, output_dir):
        self.download(output_dir)
        self.preprocess(output_dir)
        output_files = self.sharpening(output_dir)
        return output_files

    def download(self,
                 output_dir: Path,
                 parallel: bool = True,
                 download_biopar: bool = True,
                 biopar_chunk_months: int = 0,
                 s2_chunk_months: int = 0,
                 max_concurrent_jobs: int = 2) -> Dict[str, any]:
        """
        Download datacubes from specified sources.
        This method checks if data was already downloaded
        by the presence of the 'data.pkl' file and directly returns
        the output dictionary instead of reprocessing.

        Args:
            output_dir (Path): The directory where downloaded datacubes
            will be saved.
            parallel (bool, optional): Whether to download datacubes
            in parallel. Defaults to True.
            download_biopar (bool, optional): Whether to download
            biophysical variables (LAI, FAPAR, FCOVER). Set to False
            to skip BIOPAR downloads when ET computation is disabled.
            Defaults to True.
            biopar_chunk_months (int, optional): When > 0, BIOPAR jobs are
            split into chunks of this many months and submitted via
            MultiBackendJobManager. Set to 0 to use a single job per
            variable (original behaviour). Defaults to 0.
            s2_chunk_months (int, optional): When > 0, Sentinel-2 jobs are
            split into chunks of this many months and submitted via
            MultiBackendJobManager instead of a single large job. This
            reduces per-job memory and allows the default executor memory
            settings to be used. Set to 0 (default) to use a single job
            covering the full temporal extent.

        Returns:
            Dict[str, any]:

        Note:
            Format of the output dictionary:
            {   "SENTINEL2_L2A": {
                    output_path: "<Path>",
                    fmt: "<str>"
                    output_files: "<List[Path]>"
                }
                "SENTINEL3_SLSTR_L2_LST": {
                    output_path: "<Path>",
                    fmt: "<str>"
                    output_files: "<List[Path]>"
                }
                "COPERNICUS_30":
                {
                    output_path: "<Path>",
                    fmt: "<str>"
                    output_files: "<List[Path]>"
                }
                "info": {
                    "spatial_extent": {
                        "east": "<float>",
                        "north": "<float>",
                        "south": "<float>",
                        "west": "<float>"
                    },
                    "temporal_extent": [
                        "<datetime>",
                        "<datetime>"
                    ]
                }
       """
        # Path where the output_dict is pickled at the end:
        output_dict_pkl = output_dir / self.tile / f'download{self._temporal_pkl_suffix}.pkl'

        # Set output dir
        output_dir = output_dir / self.tile / '001_download'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set output paths
        output_path_s2 = output_dir / 'S2'
        output_path_s3 = (
            output_dir / 'S3'
            / f'datacube_s3{self._temporal_pkl_suffix}.nc'
        )
        output_path_dem = output_dir / 'DEM'

        # Define datacubes to download
        # Format (NAME, DATACUBE_FUNC, OUTPUT_FILE, JOBID, JOB_OPTIONS)
        datacubes = ([
            (self.name_s2,
             self._get_datacube_s2,
             output_path_s2,
             'gtiff',
             self.s2_jobid,
             type(self).JOB_OPTIONS_S2),
        ] if s2_chunk_months <= 0 else []) + [
            (self.name_s3,
             self._get_datacube_s3,
             output_path_s3,
             'netcdf',
             self.s3_jobid,
             type(self).JOB_OPTIONS_S3),

            (self.name_dem,
             self._get_datacube_dem,
             output_path_dem,
             'gtiff',
             self.dem_jobid,
             type(self).JOB_OPTIONS_DEM),

            (self.name_worldcover,
             self._get_datacube_worldcover,
             output_dir / 'WorldCover',
             'gtiff',
             self.worldcover_jobid,
             type(self).JOB_OPTIONS_WORLDCOVER),
        ] + ([
            (f'BIOPAR_{var}',
             lambda v=var: self._get_datacube_biopar(v),
             output_dir / 'BIOPAR' / var,
             'gtiff',
             None,
             type(self).JOB_OPTIONS_BIOPAR)
            for var in type(self).BIOPAR_VARIABLES
        ] if (download_biopar and biopar_chunk_months <= 0) else [])
        # If (part of) the data is already downloaded,
        # load the dictionary from the pickle file (if present),
        # then also scan output directories for any files already on disk.
        all_names = [name for name, datacube_func, output_path,
                     fmt, job_id, job_opts in datacubes]
        if output_dict_pkl.is_file():
            self._log.info('Pickle file found. Loading previous results.')
            self._download_results = self._check_and_load_pickled_dict(
                output_dict_pkl, output_dir)
        else:
            # Setup the result dict, set some initial values
            self._download_results = self._get_initial_output_dict()

        # Temporal window for filtering existing GeoTIFF files by date.
        # Only files whose embedded date falls within [t_start, t_end] are
        # considered as already-downloaded for this temporal chunk.
        t_start = datetime.strptime(
            str(self._temporal_extent[0])[:10], '%Y-%m-%d')
        t_end   = datetime.strptime(
            str(self._temporal_extent[1])[:10], '%Y-%m-%d')

        def _filter_tifs_by_extent(tif_list):
            return type(self).filter_tifs_by_extent(
                tif_list, t_start, t_end)

        # Additionally check the output directories for existing files,
        # even when they are not (yet) recorded in the pickle file.
        for name, datacube_func, output_path, fmt, job_id, job_opts in datacubes:
            if name in self._download_results:
                # Already tracked; ensure output_files list is populated and
                # that the recorded files still exist on disk (they may have
                # been deleted after the pkl was written).
                entry = self._download_results[name]
                existing = [f for f in entry.get('output_files', [])
                            if Path(f).is_file()]
                if not existing:
                    op = entry.get('output_path', output_path)
                    if fmt == 'gtiff' and op and Path(op).is_dir():
                        entry['output_files'] = _filter_tifs_by_extent(
                            sorted(Path(op).glob('*.tif')))
                    elif fmt == 'netcdf' and op and Path(op).is_file():
                        entry['output_files'] = [Path(op)]
                    elif fmt == 'netcdf':
                        # Recorded file no longer exists; try to find
                        # covering NC files from prior chunked runs.
                        nc_dir = (Path(op).parent
                                  if op and not Path(op).is_dir()
                                  else (Path(op) if op else output_path.parent))
                        covering = type(self)._find_covering_s3_netcdfs(
                            nc_dir, t_start, t_end)
                        entry['output_files'] = covering if covering else []
                    else:
                        entry['output_files'] = []
            else:
                # Not in pickle; check whether files are already on disk.
                if fmt == 'gtiff' and output_path.is_dir():
                    existing_files = _filter_tifs_by_extent(
                        sorted(output_path.glob('*.tif')))
                    if existing_files:
                        self._log.info(
                            f'{name}: found {len(existing_files)} existing '
                            f'file(s) in {output_path}. Skipping download.')
                        self._download_results[name] = {
                            'output_path': output_path,
                            'fmt': fmt,
                            'output_files': existing_files}
                elif fmt == 'netcdf' and output_path.is_file():
                    self._log.info(
                        f'{name}: found existing file {output_path}. '
                        f'Skipping download.')
                    self._download_results[name] = {
                        'output_path': output_path,
                        'fmt': fmt,
                        'output_files': [output_path]}
                elif fmt == 'netcdf' and output_path.parent.is_dir():
                    # Exact file missing – check if multiple NC files from
                    # prior chunked runs collectively cover the requested
                    # temporal extent (e.g. 3-month blocks vs full year).
                    covering = type(self)._find_covering_s3_netcdfs(
                        output_path.parent, t_start, t_end)
                    if covering:
                        self._log.info(
                            f'{name}: found {len(covering)} existing NC '
                            f'file(s) that collectively cover the requested '
                            f'temporal extent. Skipping download.')
                        self._download_results[name] = {
                            'output_path': output_path.parent,
                            'fmt': fmt,
                            'output_files': covering}

        # For chunked datasets S2 and BIOPAR are excluded from `datacubes`
        # above, so we need to check for their files on disk separately.
        if s2_chunk_months > 0:
            # Verify that any pkl-recorded S2 files still exist on disk;
            # if they have been removed, clear the stale entry so that
            # the chunked download is re-triggered below.
            if self.name_s2 in self._download_results:
                recorded_s2 = self._download_results[self.name_s2].get(
                    'output_files', [])
                if not recorded_s2 or not any(
                        Path(f).is_file() for f in recorded_s2):
                    self._log.warning(
                        f'S2: recorded output files no longer exist on disk. '
                        f'Clearing cached entry to re-trigger download.')
                    del self._download_results[self.name_s2]
            if self.name_s2 not in self._download_results:
                if output_path_s2.is_dir():
                    existing_s2 = _filter_tifs_by_extent(
                        sorted(output_path_s2.glob('*.tif')))
                    if existing_s2:
                        self._log.info(
                            f'S2: found {len(existing_s2)} existing file(s). '
                            f'Skipping chunked download.')
                        self._download_results[self.name_s2] = {
                            'output_path': output_path_s2,
                            'fmt': 'gtiff',
                            'output_files': existing_s2,
                        }

        if download_biopar and biopar_chunk_months > 0:
            for _bv in type(self).BIOPAR_VARIABLES:
                _bname = f'BIOPAR_{_bv}'
                # Verify that any pkl-recorded files still exist on disk;
                # if they have been removed, clear the stale entry so that
                # the chunked download is re-triggered below.
                if _bname in self._download_results:
                    recorded = self._download_results[_bname].get(
                        'output_files', [])
                    if not recorded or not all(
                            Path(f).is_file() for f in recorded):
                        self._log.warning(
                            f'{_bname}: one or more recorded output files no '
                            f'longer exist on disk. Clearing cached entry to '
                            f're-trigger download.')
                        del self._download_results[_bname]
                if _bname not in self._download_results:
                    _bdir = output_dir / 'BIOPAR' / _bv
                    if _bdir.is_dir():
                        existing_b = _filter_tifs_by_extent(
                            sorted(_bdir.glob('*.tif')))
                        if existing_b:
                            self._log.info(
                                f'{_bname}: found {len(existing_b)} existing '
                                f'file(s). Skipping chunked download.')
                            self._download_results[_bname] = {
                                'output_path': _bdir,
                                'fmt': 'gtiff',
                                'output_files': existing_b,
                            }

        to_download = [n for n in all_names
                       if n not in self._download_results]
        datacubes = [dc for dc in datacubes if dc[0] in to_download]

        # Determine whether any chunked job still needs to run so we do not
        # return early and skip the chunked download blocks below.
        _need_chunked_s2 = (
            s2_chunk_months > 0
            and not self._download_results.get(
                self.name_s2, {}).get('output_files'))
        _need_chunked_biopar = (
            download_biopar and biopar_chunk_months > 0
            and not all(
                self._download_results.get(
                    f'BIOPAR_{_v}', {}).get('output_files')
                for _v in type(self).BIOPAR_VARIABLES))

        if len(datacubes) == 0 and not _need_chunked_s2 \
                and not _need_chunked_biopar:
            self._log.info('All data already available. Returning results.')
            return self._download_results

        ##################################################################
        # STEP1: Process and download data from OpenEO                   #
        ##################################################################

        if parallel:
            # Download datacubes in parallel if specified
            self._log.info(
                f'Launching {len(datacubes)} openEO jobs in parallel')
            with ThreadPoolExecutor(max_workers=len(datacubes)) as executor:
                futures = list()
                for name, datacube_func, output_path, \
                        fmt, job_id, job_opts in datacubes:

                    self._log.info(f'Execute OpenEO job {name}')
                    future = executor.submit(self._execute_datacube,
                                             datacube_func,
                                             fmt,
                                             name,
                                             job_id=job_id,
                                             job_options=job_opts)
                    futures.append((name, future, output_path, fmt))

                for name, future, output_path, fmt in futures:
                    self._log.info(f'Waiting for OpenEO job {name}')
                    batch_job: BatchJob = future.result()
                    self._log.info(
                        f'OpenEO job {name} finished. Downloading results.')
                    output_files = self._download_job_result(
                        batch_job,
                        output_path,
                        output_format=fmt,
                        name=name)

                    self._download_results[name] = {
                        'output_path': output_path,
                        'fmt': fmt,
                        'output_files': output_files}

        else:
            # Download datacubes sequentially
            self._log.info(
                f'Launching {len(datacubes)} openEO jobs sequentially')
            for name, datacube, output_path, \
                    fmt, job_id, job_opts in datacubes:

                self._log.info(f'Execute OpenEO job {name}')
                batch_job: BatchJob = self._execute_datacube(
                    datacube, fmt, name, job_id, job_opts)
                self._log.info(f'OpenEO {name} finished...')
                output_files = self._download_job_result(
                    batch_job, output_path, output_format=fmt, name=name)
                self._download_results[name] = {'output_path': output_path,
                                                'fmt': fmt,
                                                'output_files': output_files}

        if self.name_s2 in to_download:
            # No scale and offsets are present in the S2 geotiffs
            # downloaded from openEO. Set them here...
            output_files_s2 = self._download_results[self.name_s2][
                'output_files']
            type(self)._add_s2_scale_offset(output_files_s2)

        # If chunked S2 download is requested, run it now via the
        # MultiBackendJobManager and merge the results into the download dict.
        if s2_chunk_months > 0:
            _s2_entry = self._download_results.get(self.name_s2)
            if _s2_entry and _s2_entry.get('output_files'):
                self._log.info(
                    'S2 already available. Skipping chunked download.')
            else:
                self._log.info(
                    f'** Running chunked S2 download '
                    f'({s2_chunk_months}-month chunks)')
                s2_chunked = self._download_s2_chunked(
                    output_dir, s2_chunk_months, max_concurrent_jobs)
                self._download_results.update(s2_chunked)
                # Apply scale/offset to the newly downloaded S2 files
                type(self)._add_s2_scale_offset(
                    self._download_results[self.name_s2]['output_files'])

        # If chunked BIOPAR download is requested, run it now via the
        # MultiBackendJobManager and merge the results into the download dict.
        if download_biopar and biopar_chunk_months > 0:
            _biopar_file_counts = {
                _v: len(self._download_results.get(
                    f'BIOPAR_{_v}', {}).get('output_files', []))
                for _v in type(self).BIOPAR_VARIABLES
            }
            _max_biopar_count = max(_biopar_file_counts.values(), default=0)
            # All BIOPAR variables are computed from the same S2 scenes, so
            # they must all produce the same number of output files. A count
            # mismatch means some chunked jobs failed and the incomplete
            # variable(s) must be re-downloaded.
            _biopar_done = (
                _max_biopar_count > 0
                and all(c == _max_biopar_count
                        for c in _biopar_file_counts.values())
            )
            if not _biopar_done:
                for _v, _c in _biopar_file_counts.items():
                    if 0 < _c < _max_biopar_count:
                        self._log.warning(
                            f'BIOPAR_{_v}: only {_c} file(s) found vs '
                            f'{_max_biopar_count} for other variable(s) — '
                            f'clearing cached entry to trigger re-download '
                            f'of missing chunks.')
                        self._download_results.pop(f'BIOPAR_{_v}', None)
            if _biopar_done:
                self._log.info(
                    'BIOPAR already available. Skipping chunked download.')
            else:
                self._log.info(
                    f'** Running chunked BIOPAR download '
                    f'({biopar_chunk_months}-month chunks)')
                biopar_chunked = self._download_biopar_chunked(
                    output_dir, biopar_chunk_months, max_concurrent_jobs)
                self._download_results.update(biopar_chunked)

        # Write the output dictionary to a pickle file
        type(self).pickle_write_dict(output_dict_pkl,
                                     output_dir, self._download_results)

        return self._download_results

    def preprocess(self, output_dir: Path, delete_unrequired_data=True):
        """ Preprocess the data
        Args:
            output_dir (Path): The output directory
            delete_unrequired_data (bool): Whether to skip data
            that is not needed in the Sharpening step,
                TODO: should be implemented properly
        Note:
            Format of output dictionary:
            {
                "SENTINEL2_L2A": {
                    "B02": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B03": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B04": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B05": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B06": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B07": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B08": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B8A": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "B11": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "NDVI": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    }
                },
                "SENTINEL3_SLSTR_L2_LST": {
                    "LST": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "LST_uncertainty": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "exception": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "confidence_in": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "sunAzimuthAngles": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "sunZenithAngles": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "viewAzimuthAngles": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "viewZenithAngles": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "confidence_in_bitlayers": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "lat": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "lon": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "sunZenithAnglesHR": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "viewAzimuthAnglesHR": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "viewZenithAnglesHR": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "latHR": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    },
                    "lonHR": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    }
                    "inc": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    }
                    "quality_flag": {
                        "<datetime>": "<Path>",
                        "...": "..."
                    }
                },
                "COPERNICUS_30": {
                    "alt": "<Path>",
                    "slo": "<Path>",
                    "asp": "<Path>"
                },
                "info": {
                    "spatial_extent": {
                        "east": "<float>",
                        "north": "<float>",
                        "south": "<float>",
                        "west": "<float>"
                    },
                    "temporal_extent": [
                        "<datetime>",
                        "<datetime>"
                    ]
                }
            }
        """

        output_dict_pkl = output_dir / self.tile / f'preprocess{self._temporal_pkl_suffix}.pkl'
        output_dir = output_dir / self.tile / '002_preprocess'

        # If the data is already preprocessed,
        # load the dictionary from the pickle file and verify that the
        # recorded files still exist on disk before skipping re-processing.
        if output_dict_pkl.is_file():
            self._preprocess_results = self._check_and_load_pickled_dict(
                output_dict_pkl, output_dir)
            missing = [
                p for p in type(self)._iter_paths(self._preprocess_results)
                if not p.is_file()
            ]
            if missing:
                self._log.warning(
                    f'Preprocess pkl found but {len(missing)} recorded '
                    f'file(s) are missing from disk '
                    f'(e.g. {missing[0].name}). Re-running preprocessing.')
                self._preprocess_results = None
            # Cache can also be stale when download() has produced new BIOPAR
            # files but an older preprocess pkl still points to a partial set.
            # If counts differ, force preprocessing to rebuild the mapping.
            elif self._download_results is not None:
                biopar_count_mismatch = []
                for _var in type(self).BIOPAR_VARIABLES:
                    _download_key = f'BIOPAR_{_var}'
                    _download_count = len(
                        self._download_results.get(
                            _download_key, {}).get('output_files', []))
                    _preprocess_count = len(
                        self._preprocess_results.get(_var, {}))
                    if _download_count != _preprocess_count:
                        biopar_count_mismatch.append(
                            (_var, _preprocess_count, _download_count))

                if biopar_count_mismatch:
                    _v, _p, _d = biopar_count_mismatch[0]
                    self._log.warning(
                        'Preprocess pkl appears stale for BIOPAR: '
                        f'{_v} has {_p} mapped file(s) in pkl but '
                        f'{_d} downloaded file(s) on disk. '
                        'Re-running preprocessing.')
                    self._preprocess_results = None
            else:
                self._log.info(
                    'Data already preprocessed. Loading results.')
                return self._preprocess_results

        # Check on download_results
        if self._download_results is None:
            raise RuntimeError('Download should be done before preprocess')

        # Get initial results
        self._preprocess_results = self._get_initial_output_dict()

        ##################################################################
        # STEP1: Convert OpenEO data, create dict structure.             #
        ##################################################################

        # - Convert format here to gtiff for intermediary data operations
        self._log.debug('Creating output dictionary')

        # Single dem in tiff format, extract extra features with richdem,
        # save as geotiff
        output_file_dem = self._download_results[self.name_dem][
            'output_files'][0]
        output_dir_dem = output_dir / 'DEM'
        self._preprocess_results[self.name_dem] = type(
            self)._extract_dem_features(output_file_dem,
                                        output_dir_dem)

        # One or more S3 netcdf files (multiple when prior chunked runs
        # already exist on disk); convert each and merge the per-timestamp
        # GeoTIFF dictionaries so that all timestamps are available.
        output_files_s3 = self._download_results[self.name_s3]['output_files']
        output_dir_s3 = output_dir / 'S3'
        s3_gtiff_merged: Dict[str, Dict[datetime, Path]] = {}
        for _nc in output_files_s3:
            _partial = type(self)._netcdf_to_geotiff_s3(
                _nc, output_dir_s3, force_f32=True)
            for _var, _date_dict in _partial.items():
                if _var not in s3_gtiff_merged:
                    s3_gtiff_merged[_var] = {}
                s3_gtiff_merged[_var].update(_date_dict)
        self._preprocess_results[self.name_s3] = s3_gtiff_merged

        # Split the different bands of the sentinel2 file. Save as geotiff
        output_files_s2 = self._download_results[self.name_s2][
            'output_files']
        output_dir_s2 = output_dir / 'S2'
        self._preprocess_results[self.name_s2] = type(
            self)._split_geotiff_s2(output_files_s2, output_dir_s2)

        ##################################################################
        # STEP2: Warp data                                               #
        ##################################################################

        # Warp to correct epsg, resolution and extent:
        # Find and validate a S2 reference file (B02).
        s2_b02_dict = self._preprocess_results[self.name_s2].get('B02', {})
        if len(s2_b02_dict) == 0:
            raise RuntimeError('No Sentinel-2 B02 files available '
                               'for preprocessing.')

        corrupt_s2_dates = set()
        s2_ref_file = None
        for _s2_date, _s2_file in list(s2_b02_dict.items()):
            try:
                with rasterio.open(_s2_file, 'r'):
                    pass
            except Exception as ex:
                self._log.warning(
                    f'Unreadable S2 B02 file detected ({_s2_file.name}): '
                    f'{ex}. Will skip timestamp {_s2_date}.')
                corrupt_s2_dates.add(_s2_date)
                continue

            if s2_ref_file is None:
                s2_ref_file = _s2_file

        if corrupt_s2_dates:
            for _band_name in list(self._preprocess_results[self.name_s2]
                                   .keys()):
                for _date in corrupt_s2_dates:
                    self._preprocess_results[self.name_s2][_band_name].pop(
                        _date, None)
            self._log.warning(
                f'Dropped {len(corrupt_s2_dates)} corrupt S2 timestamp(s): '
                + str(sorted(d.strftime('%Y%m%dT%H%M%S')
                             for d in corrupt_s2_dates)))

        if s2_ref_file is None:
            raise RuntimeError(
                'No readable Sentinel-2 B02 reference raster found. '
                'Please remove/re-download corrupted S2 files and retry.')

        var_names_hr = list()

        # Validate S3 files before warping: detect corrupt/unreadable files
        # and drop their timestamps from all S3 variables so that the rest
        # of the time series can still be processed.
        corrupt_dates = set()
        for var_name in type(self).S3_WARP_TO_HR:
            for f_in_date, f_in in list(
                    self._preprocess_results[self.name_s3].get(
                        var_name, {}).items()):
                ds = None
                try:
                    ds = gdal.Open(str(f_in))
                except RuntimeError as ex:
                    self._log.warning(
                        f'Unreadable S3 file detected ({f_in.name}): {ex}. '
                        f'Will skip timestamp {f_in_date}.')
                    corrupt_dates.add(f_in_date)
                    continue

                if ds is None:
                    self._log.warning(
                        f'Corrupt or unreadable S3 file detected, '
                        f'will skip timestamp: {f_in.name}')
                    corrupt_dates.add(f_in_date)
                ds = None
        if corrupt_dates:
            for _var in list(self._preprocess_results[self.name_s3].keys()):
                for _date in corrupt_dates:
                    self._preprocess_results[self.name_s3][_var].pop(
                        _date, None)
            self._log.warning(
                f'Dropped {len(corrupt_dates)} corrupt S3 timestamp(s): '
                + str(sorted(d.strftime('%Y%m%dT%H%M%S')
                             for d in corrupt_dates)))

        # Warp S3 HR
        # TODO: Gdalwarp does not keep scaling and offset parameters
        for var_name in type(self).S3_WARP_TO_HR:
            var_name_hr = var_name + 'HR'
            self._preprocess_results[self.name_s3][var_name_hr] = dict()

            for f_in_date, f_in in self._preprocess_results[self.name_s3][
                    var_name].items():
                f_out = f_in.parent / \
                    str(f_in.name).replace(var_name, var_name_hr)

                if not f_out.exists():
                    self._log.debug(f'Creating HiRes {f_in.name}')
                    type(self).warp_s3_to_S2(f_in, f_out, s2_ref_file, np.nan)

                self._preprocess_results[self.name_s3][var_name_hr][
                    f_in_date] = f_out
                var_names_hr.append(var_name_hr)

        # Warp other than HR to UTM
        for var_name in self._preprocess_results[self.name_s3].keys():
            if var_name not in var_names_hr:
                for f_in in self._preprocess_results[self.name_s3][
                        var_name].values():
                    self._log.debug(f'Warp to UTM {f_in.name}')
                    type(self).reference_warp_gdal(
                        f_in, ref_ds=s2_ref_file, res=self._s3_pixel_size)

        # Warp dem data
        for var_name, f_in in self._preprocess_results[self.name_dem].items():
            type(self).reference_warp_gdal(f_in, ref_ds=s2_ref_file)

        # Preprocess biopar: copy to preprocess dir, then align to S2 grid.
        # Files are copied first (like WorldCover) so that:
        #   - the original download files are never modified, and
        #   - re-running preprocess is idempotent.
        for var in type(self).BIOPAR_VARIABLES:
            key = f'BIOPAR_{var}'
            if key in self._download_results:
                biopar_files = self._download_results[key]['output_files']
                biopar_src_dict = type(self)._create_output_dict_gtiff(
                    biopar_files)
                biopar_dst_dir = output_dir / 'BIOPAR' / var
                biopar_dst_dir.mkdir(parents=True, exist_ok=True)
                biopar_date_dict = dict()
                self._log.info(
                    f'Copying and aligning {var} biopar to S2 reference grid')
                for f_date, f_src in biopar_src_dict.items():
                    f_dst = biopar_dst_dir / f_src.name
                    if not f_dst.exists():
                        shutil.copy2(f_src, f_dst)
                    type(self).reference_warp_gdal(f_dst, ref_ds=s2_ref_file)
                    biopar_date_dict[f_date] = f_dst
                self._preprocess_results[var] = biopar_date_dict

        # Preprocess WorldCover: warp single GeoTIFF to S2 20m reference grid
        if self.name_worldcover in self._download_results:
            wc_files = self._download_results[self.name_worldcover][
                'output_files']
            if wc_files:
                wc_src = wc_files[0]
                wc_dst = (output_dir / 'WorldCover'
                          / wc_src.name)
                wc_dst.parent.mkdir(parents=True, exist_ok=True)
                if not wc_dst.exists():
                    import shutil as _shutil
                    _shutil.copy2(wc_src, wc_dst)
                self._log.info(
                    f'Warping WorldCover to S2 reference grid ({wc_dst.name})')
                type(self).reference_warp_gdal(
                    wc_dst, ref_ds=s2_ref_file, resampling='mode')
                self._preprocess_results[self.name_worldcover] = wc_dst

        ##################################################################
        # STEP3: Generate quality_flags                                  #
        ##################################################################

        self._preprocess_results[self.name_s3]['quality_flag'] \
            = self._generate_quality_flag(self._preprocess_results)

        ##################################################################
        # STEP4: Calculate incidence angles (inc, cos_theta)             #
        ##################################################################

        self._preprocess_results[self.name_s3]['inc'] \
            = self._calculate_incidence_angle(self._preprocess_results)

        ##################################################################
        # STEP5: Cleanup unrequired data.                                #
        ##################################################################
        # Remove unnecessary data to save disk space and resources
        # before the sharpening step
        if delete_unrequired_data:
            s3_required = ('LST', 'inc', 'quality_flag',
                           'viewZenithAnglesHR', 'latHR', 'lonHR')
            s2_required = type(self).S2_SHARP_INPUTS
            s2_required.append('NDVI')
            dem_required = ('alt',)

            s3_dict = self._preprocess_results[self.name_s3]
            s3_keys = list(s3_dict.keys())

            for var_name in s3_keys:
                if var_name not in s3_required:
                    type(self).remove_key_and_file(s3_dict, var_name)

            s2_dict = self._preprocess_results[self.name_s2]
            s2_keys = list(s2_dict.keys())

            for var_name in s2_keys:
                if var_name not in s2_required:
                    type(self).remove_key_and_file(s2_dict, var_name)

            dem_dict = self._preprocess_results[self.name_dem]
            dem_keys = list(dem_dict.keys())

            for var_name in dem_keys:
                if var_name not in dem_required:
                    type(self).remove_key_and_file(dem_dict, var_name)

        ##################################################################
        # STEP 6: Convert to correct data format                         #
        ##################################################################
        scale = True  # Temporary disable scaling.
        if scale:
            scaling_data = type(self).get_scaling_data()
            s3_required_for_pipeline = {
                'LST', 'inc', 'quality_flag',
                'viewZenithAnglesHR', 'latHR', 'lonHR'
            }

            # Do S3
            for var_name in list(self._preprocess_results[self.name_s3].keys()):
                for f_in_date, f_in in list(
                        self._preprocess_results[self.name_s3][
                            var_name].items()):
                    var_scaling_data = scaling_data[self.name_s3].get(
                        var_name, None)
                    if var_scaling_data is not None:
                        self._log.debug(f'Apply S3 Scale/offset {f_in.name}')
                        try:
                            type(self).to_scaled_raster(
                                f_in, **var_scaling_data)
                        except rasterio.errors.RasterioIOError as ex:
                            if var_name in s3_required_for_pipeline:
                                raise RuntimeError(
                                    f'Corrupt required S3 raster: {f_in}. '
                                    'Please remove/re-download this scene '
                                    'and rerun preprocessing.') from ex

                            self._log.warning(
                                f'Skipping unreadable optional S3 raster '
                                f'{f_in.name} ({var_name}, '
                                f'{f_in_date}): {ex}')
                            try:
                                if f_in.exists():
                                    f_in.unlink()
                            except Exception:
                                pass
                            self._preprocess_results[self.name_s3][
                                var_name].pop(f_in_date, None)

            # Do DEM
            for var_name in self._preprocess_results[self.name_dem].keys():
                f_in = self._preprocess_results[self.name_dem][var_name]
                var_scaling_data = scaling_data[self.name_dem][var_name]
                if var_scaling_data is not None:
                    self._log.debug(f'Apply DEM Scale/offset {f_in.name}')
                    type(self).to_scaled_raster(f_in, **var_scaling_data)

        #################################################################
        # STEP5: Return Output Dictionary                               #
        #################################################################
        type(self).pickle_write_dict(output_dict_pkl,
                                     output_dir, self._preprocess_results)

        return self._preprocess_results

    def sharpening(self, output_dir: Path,
                   residual_correction: bool = False,
                   min_valid_fraction: float = 0.0,
                   mask_to_s3_coverage: bool = False):

        # Build a suffix encoding the filtering/masking parameters so that
        # runs with different settings produce distinct output files and
        # pickles and do not collide with each other.
        _minf_pct = int(round(min_valid_fraction * 100))
        _file_suffix = f'_minf{_minf_pct:03d}'
        if mask_to_s3_coverage:
            _file_suffix += '_msk'

        output_dict_pkl = output_dir / self.tile / f'sharpening{self._temporal_pkl_suffix}{_file_suffix}.pkl'
        output_dir = output_dir / self.tile / '003_sharpening'
        tmp_dir = output_dir / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if output_dict_pkl.is_file():
            self._sharpening_results = self._check_and_load_pickled_dict(
                output_dict_pkl, output_dir)
            self._log.info('Data already preprocessed. Loading results.')
            return self._sharpening_results

        if self._preprocess_results is None:
            raise RuntimeError('Preprocessor should run before sharpening')

        self._sharpening_results = self._get_initial_output_dict()

        s2_data: dict = self._preprocess_results[self.name_s2]
        s3_data: dict = self._preprocess_results[self.name_s3]
        dem_data: dict = self._preprocess_results[self.name_dem]

        s2_dates: List[datetime] = list(s2_data['B02'].keys())
        s3_dates: List[datetime] = list(s3_data['LST'].keys())

        # Store output files here
        output_files: Dict[datetime, Path] = dict()

        # for each tile
        # get list of s3 dates --> s3_dates
        # get list of available s2 observations --> s2_dates
        # for each s3 date, get closest available s2 observation
        for s3_date in s3_dates:
            output_file = output_dir / \
                f'LST_SHARPENED_{s3_date.strftime("%Y%m%dT%H%M%S")}{_file_suffix}.tif'
            output_fin = Path(str(output_file).replace('.tif', '_fin.tif'))

            # Stable reference to the original S3 LST — do NOT use lowres_file
            # for this purpose because it gets overwritten inside the
            # BlockHelper loop further below.
            s3_lst_file = s3_data['LST'][s3_date]

            if output_fin.is_file():
                output_files[s3_date] = output_fin
                self._log.info(f'{s3_date} already processed. ({output_fin})')
                if mask_to_s3_coverage:
                    self._log.info(
                        f'{s3_date}: (Re-)applying S3 coverage mask '
                        f'to existing file.')
                    with rasterio.open(s3_lst_file) as _s3:
                        _s3_arr = _s3.read(1)
                        _s3_nd = (
                            _s3.nodata if _s3.nodata is not None else 0)
                        _s3_valid = (
                            _s3_arr != _s3_nd).astype(np.uint8)
                        _s3_tr = _s3.transform
                        _s3_crs = _s3.crs
                    with rasterio.open(output_fin) as _fin:
                        _fin_profile = _fin.profile.copy()
                        _fin_scales = _fin.scales
                        _fin_descriptions = _fin.descriptions
                        _fin_units = _fin.units
                        _fin_data = _fin.read(1)
                        _fin_nodata = _fin.nodata
                        _fin_tr = _fin.transform
                        _fin_crs = _fin.crs
                        _fin_h = _fin.height
                        _fin_w = _fin.width
                    _s3_valid_hr = np.zeros(
                        (_fin_h, _fin_w), dtype=np.uint8)
                    rasterio.warp.reproject(
                        source=_s3_valid,
                        destination=_s3_valid_hr,
                        src_transform=_s3_tr,
                        src_crs=_s3_crs,
                        dst_transform=_fin_tr,
                        dst_crs=_fin_crs,
                        resampling=Resampling.nearest,
                    )
                    _fin_data[_s3_valid_hr == 0] = _fin_nodata
                    with rasterio.open(
                            output_fin, 'w', **_fin_profile) as _dst:
                        _dst.scales = _fin_scales
                        _dst.descriptions = _fin_descriptions
                        _dst.units = _fin_units
                        _dst.write(_fin_data, 1)
                        _dst.build_overviews(
                            (4, 8, 16), Resampling.average)
                continue

            # start processing...
            # Early-exit if the LST raster is entirely nodata / empty
            # (e.g. fully clouded scene) to avoid wasting time building
            # the VRT and training the sharpener on zero pixels.
            # Also skip if the fraction of valid S3 pixels is below the
            # user-defined minimum threshold (min_valid_fraction).
            lowres_file = s3_data['LST'][s3_date]
            with rasterio.open(lowres_file) as _chk:
                _lst_chk = _chk.read(1)
                _nd = _chk.nodata if _chk.nodata is not None else 0
            _valid_frac = np.sum(_lst_chk != _nd) / _lst_chk.size
            if _valid_frac == 0.0:
                self._log.warning(
                    f'{s3_date}: LST raster is entirely nodata — skipping.')
                continue
            if _valid_frac < min_valid_fraction:
                self._log.warning(
                    f'{s3_date}: LST raster has only '
                    f'{_valid_frac:.1%} valid pixels '
                    f'(min_valid_fraction={min_valid_fraction:.1%}) '
                    f'— skipping.')
                continue

            s2_date_str = find_closest_date(s3_date, s2_dates)
            s2_date = datetime.strptime(s2_date_str, '%Y%m%d')

            # now, all Sentinel-2 bands + DEM altitude
            # + cos_theta should be combined in one VRT file,
            # representing the high resolution inputs to the model
            highres_inputs = list()
            for band in type(self).S2_SHARP_INPUTS:
                highres_inputs.append(str(s2_data[band][s2_date]))

            highres_inputs.append(str(dem_data['alt']))
            highres_inputs.append(str(s3_data['inc'][s3_date]))

            vrt_filename = tmp_dir / \
                f'{s3_date.strftime("%Y%m%dT%H%M%S")}.vrt'

            my_vrt = gdal.BuildVRT(str(vrt_filename), highres_inputs,
                                   separate=True)
            my_vrt = None

            lowres_file = s3_data['LST'][s3_date]
            highres_file = vrt_filename
            # For some reason, the trainer adds one pixel
            # in x and y direction in the dataset.
            # Make a copy of the lowres file
            lowres_file_copy = tmp_dir / \
                f'LST_COPY-{s3_date.strftime("%Y%m%dT%H%M%S")}.tif'
            shutil.copyfile(lowres_file, lowres_file_copy)

            mask_files = [str(s3_data['quality_flag'][s3_date])]
            lst_good_quality_flags = [1]
            cv_homogeneity_threshold = 0.8
            moving_window_size = 3

            dms_options = \
                {"highResFiles": [str(highres_file)],
                    "lowResFiles": [str(lowres_file_copy)],
                    "highresbandnames": type(self).SHARPENING_INPUTS,
                    "lowResQualityFiles": mask_files,
                    "lowResGoodQualityFlags": lst_good_quality_flags,
                    "cvHomogeneityThreshold": cv_homogeneity_threshold,
                    "movingWindowSize": moving_window_size,
                    "disaggregatingTemperature": True,
                    "baggingRegressorOpt": {
                        "n_jobs": (
                            1 if sys.gettrace() is not None
                            else SenETDownload.NUM_JOBS_DISAGGREGATOR),
                        "n_estimators": 30,
                        "max_samples": 0.8,
                        "max_features": 0.8}}

            disaggregator = DecisionTreeSharpener(**dms_options)

            self._log.info(f'lstsharpener train: {s3_date}')
            try:
                disaggregator.trainSharpener()
            except ValueError as e:
                self._log.warning(
                    f'{s3_date} could not be trained '
                    f'(maybe the S3 grid is empty?): {str(e)} ')
            else:
                # Once trained successfully,
                # we apply the sharpener to the full tile...

                # Use B02 as reference for output dataset creation
                highres_handler: rasterio.io.DatasetReader
                with rasterio.open(s2_data['B02'][s2_date],
                                   'r') as highres_handler:
                    output_profile = highres_handler.profile.copy()

                scaling_data = type(self).get_scaling_data()[
                    'SENTINEL3_SLSTR_L2_LST']['LST']
                output_profile['dtype'] = scaling_data['dtype']
                output_profile['nodata'] = scaling_data['nodata']
                output_profile['count'] = 1
                output_profile['compress'] = 'deflate'
                output_profile['blockxsize'] = 256
                output_profile['blockysize'] = 256

                output_handler: rasterio.io.DatasetWriter
                with rasterio.open(
                    output_file, 'w', **output_profile,
                    num_threads=SenETDownload.NUM_THREADS_RIO) \
                        as output_handler:

                    output_handler.scales = [scaling_data['scale']]
                    output_handler.descriptions = ['LST_SHARPENED']
                    output_handler.units = ['K']

                    tmp_dir_block_helper = tmp_dir / \
                        f'block_helper_{s3_date.strftime("%Y%m%dT%H%M%S")}'
                    with BlockHelper(lowres_file,
                                     highres_file,
                                     low_res_x_block_size=35,
                                     low_res_y_block_size=35,
                                     tmp_dir=tmp_dir_block_helper,
                                     cleanup=True) as block_helper:

                        for i in range(0, block_helper.window_count):
                            lowres_file = block_helper.get_lo_res_file(i)
                            highres_file = block_helper.get_hi_res_file(i)

                            # memory = type(
                            #     self).get_current_memory_usage(
                            # ).vms / 1024 / 1024
                            self._log.info(f'lstsharpener apply: '
                                           f'{s3_date} - '
                                           f'(Block: {i + 1}/'
                                           f'{block_helper.window_count})'
                                           #    ' - '
                                           #    f'(Memory usage: {memory} MiB)'
                                           )

                            output_f32 = disaggregator.applySharpener(
                                str(highres_file), str(lowres_file))

                            # Apply scaling.
                            output_f32 = output_f32 / scaling_data['scale']
                            output_f32 = np.nan_to_num(
                                output_f32, scaling_data['nodata'])
                            output_scaled = output_f32.astype(
                                np.dtype(scaling_data['dtype']))

                            window = block_helper.get_hi_res_window(i)
                            output_handler.write(
                                output_scaled, 1, window=window)

                    # Generate overviews
                    output_handler.build_overviews(
                        (4, 8, 16), Resampling.average)

                if residual_correction:
                    # Now run corrections based on residual analysis
                    maskfile = mask_files[0]
                    lowres_file = s3_data['LST'][s3_date]
                    ri, ci = disaggregator.residualAnalysis(
                        str(output_file), str(lowres_file),
                        maskfile, doCorrection=True)
                    lst = ci.GetRasterBand(1).ReadAsArray()
                    res = ri.GetRasterBand(1).ReadAsArray()
                    # Apply scaling.
                    lst = lst / scaling_data['scale']
                    lst = np.nan_to_num(
                        lst, scaling_data['nodata'])
                    lst = lst.astype(
                        np.dtype(scaling_data['dtype']))

                    with rasterio.open(
                            output_fin, 'w',
                            **output_profile) as output_handler:
                        output_handler.scales = [scaling_data['scale']]
                        output_handler.descriptions = ['LST_SHARPENED_FIN']
                        output_handler.units = ['K']
                        output_handler.write(lst, 1)
                        # Generate overviews
                        output_handler.build_overviews(
                            (4, 8, 16), Resampling.average)

                    residuals_file = str(output_file).replace(
                        '.tif', '_residuals.tif')
                    with rasterio.open(
                            residuals_file, 'w',
                            **output_profile) as output_handler:
                        output_handler.descriptions = ['LST_RESIDUALS']
                        output_handler.write(res, 1)

                else:
                    # use non-corrected file as final output
                    os.rename(output_file, output_fin)

                if mask_to_s3_coverage:
                    # Mask sharpened LST to only pixels where S3 had valid
                    # data by reprojecting the S3 valid mask to the output
                    # high-res grid and applying it.
                    # Use s3_lst_file (not lowres_file_copy or lowres_file)
                    # because lowres_file was overwritten in the block loop.
                    self._log.info(
                        f'{s3_date}: Masking sharpened LST to S3 coverage.')
                    with rasterio.open(s3_lst_file) as _s3:
                        _s3_arr = _s3.read(1)
                        _s3_nd = (_s3.nodata
                                  if _s3.nodata is not None else 0)
                        _s3_valid = (_s3_arr != _s3_nd).astype(np.uint8)
                        _s3_tr = _s3.transform
                        _s3_crs = _s3.crs
                    with rasterio.open(output_fin) as _fin:
                        _fin_profile = _fin.profile.copy()
                        _fin_data = _fin.read(1)
                        _fin_nodata = _fin.nodata
                        _fin_tr = _fin.transform
                        _fin_crs = _fin.crs
                        _fin_h = _fin.height
                        _fin_w = _fin.width
                    _s3_valid_hr = np.zeros(
                        (_fin_h, _fin_w), dtype=np.uint8)
                    rasterio.warp.reproject(
                        source=_s3_valid,
                        destination=_s3_valid_hr,
                        src_transform=_s3_tr,
                        src_crs=_s3_crs,
                        dst_transform=_fin_tr,
                        dst_crs=_fin_crs,
                        resampling=Resampling.nearest,
                    )
                    _fin_data[_s3_valid_hr == 0] = _fin_nodata
                    with rasterio.open(
                            output_fin, 'w', **_fin_profile) as _dst:
                        _dst.scales = [scaling_data['scale']]
                        _dst.descriptions = ['LST_SHARPENED_FIN']
                        _dst.units = ['K']
                        _dst.write(_fin_data, 1)
                        _dst.build_overviews(
                            (4, 8, 16), Resampling.average)

                output_files[s3_date] = output_fin

            disaggregator = None
            gc.collect()

        # Final results to pkl
        self._sharpening_results['LST'] = output_files
        type(self).pickle_write_dict(output_dict_pkl,
                                     output_dir, self._sharpening_results)

        return self._sharpening_results

    def lst_correction(self, output_dir: Path,
                       corr_parameters: dict = None) -> dict:
        """_summary_

        Parameters
        ----------
        output_dir : Path
            _description_
        sharpening_dict : dict
            _description_
        corr_parameters : dict, optional
            _description_, by default None

        Returns
        -------
        dict
            _description_
        """

        lst_dict_pkl = output_dir / self.tile / f'lst_correction{self._temporal_pkl_suffix}.pkl'

        if corr_parameters is None:
            # No correction required
            self._lst_results = self._sharpening_results.copy()
            sharpening_pkl = output_dir / self.tile / f'sharpening{self._temporal_pkl_suffix}.pkl'
            shutil.copyfile(sharpening_pkl, lst_dict_pkl)

        else:

            output_dir = output_dir / self.tile / '004_lst-correction'
            output_dir.mkdir(parents=True, exist_ok=True)

            if lst_dict_pkl.is_file():
                self._lst_results = self._check_and_load_pickled_dict(
                    lst_dict_pkl, output_dir)
                self._log.info('Data already corrected. Loading results.')
                return self._lst_results

            if self._sharpening_results is None:
                raise RuntimeError(
                    'Sharpening should run before LST correction')

            self._lst_results = self._get_initial_output_dict()

            dates: List[datetime] = list(
                self._sharpening_results['LST'].keys())

            # Store output files here
            output_files: Dict[datetime, Path] = dict()

            for date in dates:

                output_file = output_dir / \
                    f'LST_CORRECTED_{date.strftime("%Y%m%dT%H%M%S")}.tif'
                if output_file.is_file():

                    self._log.info(
                        f'{date} already processed. ({output_file})')
                else:
                    lst_file = self._sharpening_results['LST'][date]

                    s3_dict = self._preprocess_results[self.name_s3]
                    vza_file = s3_dict['viewZenithAnglesHR'][date]

                    # Apply LST correction
                    self._log.info(f'lst correction apply: {date}')
                    _lst_correction(lst_file, vza_file, output_file,
                                    **corr_parameters)

                output_files[date] = output_file

            self._lst_results['LST'] = output_files
            type(self).pickle_write_dict(lst_dict_pkl, output_dir,
                                         self._lst_results)

        return self._lst_results

    def _download_s2_chunked(
            self,
            output_dir: Path,
            chunk_months: int = 1,
            max_concurrent_jobs: int = 2) -> dict:
        """Download Sentinel-2 data as monthly temporal chunks via
        ``openeo.extra.job_management.MultiBackendJobManager``.

        Submits one OpenEO batch job per temporal chunk, tracks them via a
        persistent CSV job database (resumable if interrupted), then collects
        all downloaded GeoTIFFs into the standard ``S2/`` directory used by
        ``preprocess()``.

        Args:
            output_dir (Path): The ``001_download`` sub-directory.
            chunk_months (int): Number of months per temporal chunk.

        Returns:
            dict: Keyed by ``self.name_s2`` with the standard
            download-result structure
            ``{'output_path': Path, 'fmt': 'gtiff', 'output_files': [...]``.
        """
        try:
            from openeo.extra.job_management import (
                MultiBackendJobManager,
                CsvJobDatabase,
            )
        except ImportError as exc:
            raise ImportError(
                'openeo.extra.job_management is required for chunked S2 '
                'download (openeo-python-client >= 0.31.0).'
            ) from exc

        # ── build temporal chunks ──────────────────────────────────────────
        start_dt = datetime.strptime(self._temporal_extent[0], '%Y-%m-%d')
        end_dt   = datetime.strptime(self._temporal_extent[1], '%Y-%m-%d')
        chunks: List[tuple] = []
        current = start_dt
        while current <= end_dt:
            m         = current.month - 1 + chunk_months
            next_dt   = datetime(current.year + m // 12, m % 12 + 1, 1)
            chunk_end = min(next_dt - timedelta(days=1), end_dt)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d'),
            ))
            current = next_dt
        self._log.info(
            f'S2 chunked download: {len(chunks)} chunks')

        s2_dir = output_dir / 'S2'
        s2_dir.mkdir(parents=True, exist_ok=True)

        # ── build jobs DataFrame ───────────────────────────────────────────
        jobs_df = pd.DataFrame(
            [{'start_date': cs, 'end_date': ce} for cs, ce in chunks])

        # ── job manager setup ──────────────────────────────────────────────
        manager_root = output_dir / 'S2' / '_job_manager'
        manager_root.mkdir(parents=True, exist_ok=True)
        job_db_path = (
            output_dir.parent
            / f's2_jobs{self._temporal_pkl_suffix}.csv'
        )

        eoconn         = openeo.connect(
            type(self).OPENEO_CDSE_URL).authenticate_oidc()
        spatial_extent = self._spatial_extent
        s2_should_mask        = self._s2_should_mask
        s2_should_composite   = self._s2_should_composite
        s2_should_interpolate = self._s2_should_interpolate
        name_s2         = self.name_s2
        tile            = self.tile
        job_options     = type(self).JOB_OPTIONS_S2
        band_names      = type(self).S2_BAND_NAMES

        def start_job(row, connection, **kwargs):
            chunk_extent = [row['start_date'], row['end_date']]
            cloud_props = {'eo:cloud_cover': lambda val: val <= 95.0}
            cube = connection.load_collection(
                name_s2,
                temporal_extent=chunk_extent,
                spatial_extent=spatial_extent,
                bands=band_names,
                properties=cloud_props,
            ).resample_spatial(resolution=20)

            if s2_should_mask:
                scl_cube = connection.load_collection(
                    collection_id=name_s2,
                    bands=['SCL'],
                    temporal_extent=chunk_extent,
                    spatial_extent=spatial_extent,
                    properties=cloud_props,
                )
                scl_dilated_mask = scl_cube.process(
                    'to_scl_dilation_mask',
                    data=scl_cube,
                    scl_band_name='SCL',
                    kernel1_size=17,
                    kernel2_size=201,
                    mask1_values=[2, 4, 5, 6, 7],
                    mask2_values=[3, 8, 9, 10, 11],
                    erosion_kernel_size=3,
                ).rename_labels('bands', ['S2-L2A-SCL_DILATED_MASK'])
                cube = cube.mask(scl_dilated_mask)

            cube = cube.ndvi(nir='B08', red='B04', target_band='NDVI')

            if s2_should_composite:
                cube = cube.aggregate_temporal_period(
                    period='dekad', reducer='median')

            if s2_should_interpolate:
                cube = cube.apply_dimension(
                    dimension='t',
                    process='array_interpolate_linear')

            return cube.create_job(
                out_format='gtiff',
                title=f'S2_{tile}_{row["start_date"]}_{row["end_date"]}',
                job_options=job_options,
            )

        manager = MultiBackendJobManager(
            root_dir=str(manager_root),
        )
        manager.add_backend('cdse', connection=eoconn,
                            parallel_jobs=max_concurrent_jobs)
        manager.run_jobs(
            df=jobs_df,
            start_job=start_job,
            job_db=CsvJobDatabase(job_db_path),
        )

        # ── move downloaded files to S2 dir & report failures ───────────────
        job_db_df = pd.read_csv(job_db_path)
        failed = job_db_df[
            job_db_df['status'].isin(['error', 'cancelled'])]
        if not failed.empty:
            self._log.warning(
                f'S2 chunked: {len(failed)} job(s) failed/cancelled:\n'
                + failed[['start_date', 'end_date',
                           'status']].to_string(index=False))

        for _, row in job_db_df[
                job_db_df['status'] == 'finished'].iterrows():
            job_dir = manager_root / f"job_{row['id']}"
            for f in sorted(job_dir.rglob('*.tif')):
                # Rename openEO_<date>Z.tif → SENTINEL2_L2A_<date>Z.tif
                dest_name = re.sub(r'^openEO', self.name_s2, f.name)
                dest = s2_dir / dest_name
                if not dest.exists():
                    shutil.move(str(f), str(dest))

        # ── build standard download-result structure ───────────────────────
        t_start = datetime.strptime(
            str(self._temporal_extent[0])[:10], '%Y-%m-%d')
        t_end = datetime.strptime(
            str(self._temporal_extent[1])[:10], '%Y-%m-%d')
        output_files = type(self).filter_tifs_by_extent(
            sorted(s2_dir.glob('*.tif')), t_start, t_end)
        self._log.info(
            f'S2 chunked: {len(output_files)} file(s) collected in {s2_dir}')
        return {
            self.name_s2: {
                'output_path':  s2_dir,
                'fmt':          'gtiff',
                'output_files': output_files,
            }
        }

    def _download_biopar_chunked(
            self,
            output_dir: Path,
            chunk_months: int = 1,
            max_concurrent_jobs: int = 2) -> dict:
        """Download BIOPAR variables as monthly temporal chunks via
        ``openeo.extra.job_management.MultiBackendJobManager``.

        Submits one OpenEO batch job per (variable, chunk) combination,
        tracks them via a persistent CSV job database (resumable if
        interrupted), then moves the downloaded GeoTIFFs into the standard
        per-variable subdirectory layout used by ``preprocess()``.

        Args:
            output_dir (Path): The ``001_download`` sub-directory.
            chunk_months (int): Number of months per temporal chunk.

        Returns:
            dict: Keyed by ``'BIOPAR_{var}'`` with the standard
            download-result structure
            ``{'output_path': Path, 'fmt': 'gtiff', 'output_files': [...]``.
        """
        try:
            from openeo.extra.job_management import (
                MultiBackendJobManager,
                CsvJobDatabase,
            )
        except ImportError as exc:
            raise ImportError(
                'openeo.extra.job_management is required for chunked BIOPAR '
                'download (openeo-python-client >= 0.31.0).'
            ) from exc

        variables = type(self).BIOPAR_VARIABLES

        # ── build temporal chunks ──────────────────────────────────────────
        start_dt = datetime.strptime(self._temporal_extent[0], '%Y-%m-%d')
        end_dt   = datetime.strptime(self._temporal_extent[1], '%Y-%m-%d')
        chunks: List[tuple] = []
        current = start_dt
        while current <= end_dt:
            m         = current.month - 1 + chunk_months
            next_dt   = datetime(current.year + m // 12, m % 12 + 1, 1)
            chunk_end = min(next_dt - timedelta(days=1), end_dt)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d'),
            ))
            current = next_dt
        self._log.info(
            f'BIOPAR chunked download: {len(variables)} vars \u00d7 '
            f'{len(chunks)} chunks = {len(variables) * len(chunks)} jobs')

        # ── create per-variable output directories ─────────────────────────
        for var in variables:
            (output_dir / 'BIOPAR' / var).mkdir(parents=True, exist_ok=True)

        # ── build jobs DataFrame ───────────────────────────────────────────
        rows = [
            {'biopar_var': var, 'start_date': cs, 'end_date': ce}
            for var in variables
            for cs, ce in chunks
        ]
        jobs_df = pd.DataFrame(rows)

        # ── job manager setup ──────────────────────────────────────────────
        # Job results are downloaded to manager_root/<job_id>/
        manager_root = output_dir / 'BIOPAR' / '_job_manager'
        manager_root.mkdir(parents=True, exist_ok=True)
        # Job DB CSV lives next to the other pkl files in the tile directory
        job_db_path = (
            output_dir.parent
            / f'biopar_jobs{self._temporal_pkl_suffix}.csv'
        )

        eoconn = openeo.connect(
            type(self).OPENEO_CDSE_URL).authenticate_oidc()
        spatial_extent = self._spatial_extent
        biopar_udp_url = type(self).BIOPAR_UDP_URL
        job_options    = type(self).JOB_OPTIONS_BIOPAR
        tile           = self.tile

        def start_job(row, connection, **kwargs):
            cube = connection.datacube_from_process(
                process_id='biopar',
                namespace=biopar_udp_url,
                spatial_extent=spatial_extent,
                temporal_extent=[row['start_date'], row['end_date']],
                biopar_type=row['biopar_var'],
            )
            return cube.create_job(
                out_format='gtiff',
                title=(
                    f"biopar_{row['biopar_var']}_{tile}"
                    f"_{row['start_date']}_{row['end_date']}"
                ),
                job_options=job_options,
            )

        manager = MultiBackendJobManager(
            root_dir=str(manager_root),
        )
        manager.add_backend('cdse', connection=eoconn,
                            parallel_jobs=max_concurrent_jobs)

        # Retry transient failed/cancelled chunks instead of silently
        # continuing with partial BIOPAR coverage.
        pending_df = jobs_df.copy()
        max_attempts = type(self).BIOPAR_CHUNK_RETRIES + 1
        attempt = 1
        while True:
            self._log.info(
                f'BIOPAR chunked submission attempt '
                f'{attempt}/{max_attempts}: {len(pending_df)} job(s)')
            manager.run_jobs(
                df=pending_df,
                start_job=start_job,
                job_db=CsvJobDatabase(job_db_path),
            )

            job_db_df = pd.read_csv(job_db_path)
            failed = job_db_df[
                job_db_df['status'].isin(['error', 'cancelled'])]
            if failed.empty:
                break

            self._log.warning(
                f'BIOPAR chunked: {len(failed)} job(s) failed/cancelled '
                f'after attempt {attempt}/{max_attempts}:\n'
                + failed[['biopar_var', 'start_date',
                           'end_date', 'status']].to_string(index=False))

            if attempt >= max_attempts:
                raise RuntimeError(
                    'BIOPAR chunked download exhausted retries with '
                    f'{len(failed)} failed/cancelled job(s). '
                    'Please rerun, or inspect backend/job logs.')

            pending_df = failed[
                ['biopar_var', 'start_date', 'end_date']
            ].drop_duplicates().reset_index(drop=True)
            attempt += 1
            self._log.info(
                f'Retrying failed BIOPAR chunks in '
                f'{type(self).RETRY_DELAY} seconds...')
            time.sleep(type(self).RETRY_DELAY)

        # ── move downloaded files to per-variable dirs ─────────────────────
        job_db_df = pd.read_csv(job_db_path)

        for _, row in job_db_df[
                job_db_df['status'] == 'finished'].iterrows():
            job_dir = manager_root / f"job_{row['id']}"
            var_dir = output_dir / 'BIOPAR' / row['biopar_var']
            bname = f"BIOPAR_{row['biopar_var']}"
            for f in sorted(job_dir.rglob('*.tif')):
                # Rename openEO_<date>Z.tif → BIOPAR_<VAR>_<date>Z.tif
                dest_name = re.sub(r'^openEO', bname, f.name)
                dest = var_dir / dest_name
                if not dest.exists():
                    shutil.move(str(f), str(dest))

        # ── build standard download-result structure ───────────────────────
        b_t_start = datetime.strptime(
            str(self._temporal_extent[0])[:10], '%Y-%m-%d')
        b_t_end = datetime.strptime(
            str(self._temporal_extent[1])[:10], '%Y-%m-%d')
        results: dict = {}
        for var in variables:
            var_dir      = output_dir / 'BIOPAR' / var
            output_files = type(self).filter_tifs_by_extent(
                sorted(var_dir.glob('*.tif')), b_t_start, b_t_end)
            results[f'BIOPAR_{var}'] = {
                'output_path': var_dir,
                'fmt':         'gtiff',
                'output_files': output_files,
            }
            self._log.info(
                f'BIOPAR {var}: {len(output_files)} file(s) collected')
        return results

    def compute_biopar(self,
                       output_dir: Path,
                       variables: List[str] = None) -> dict:
        """Compute biophysical variables (LAI, FAPAR, FCOVER) using the
        ESA-APEx biopar UDP on CDSE OpenEO. One OpenEO job is launched
        per variable since the UDP only computes one variable at a time.

        Args:
            output_dir (Path): Root output directory.
            variables (List[str], optional): List of variables to compute.
                Defaults to BIOPAR_VARIABLES = ["LAI", "FAPAR", "FCOVER"].

        Returns:
            dict: {
                "LAI":   {<datetime>: <Path>, ...},
                "FAPAR": {<datetime>: <Path>, ...},
                "FCOVER":{<datetime>: <Path>, ...}
            }
        """
        if variables is None:
            variables = type(self).BIOPAR_VARIABLES

        biopar_pkl = output_dir / self.tile / f'biopar{self._temporal_pkl_suffix}.pkl'
        biopar_dir = output_dir / self.tile / '005_biopar'
        biopar_dir.mkdir(parents=True, exist_ok=True)

        if biopar_pkl.is_file():
            self._log.info('Biopar already computed. Loading results.')
            self._biopar_results = self._check_and_load_pickled_dict(
                biopar_pkl, biopar_dir)
            return self._biopar_results

        self._biopar_results = self._get_initial_output_dict()

        eoconn = openeo.connect(
            type(self).OPENEO_CDSE_URL).authenticate_oidc()

        # Run one OpenEO job per variable
        for biopar_var in variables:
            var_dir = biopar_dir / biopar_var
            var_dir.mkdir(parents=True, exist_ok=True)

            self._log.info(f'Computing biopar variable: {biopar_var}')

            # The biopar UDP handles S2 loading (B03, B04, B08 + angle bands),
            # band selection, and SCL cloud masking internally.
            # Pass spatial_extent, temporal_extent, and biopar_type as
            # top-level parameters — do NOT pass a pre-built data cube.
            biopar_cube = eoconn.datacube_from_process(
                process_id="biopar",
                namespace=type(self).BIOPAR_UDP_URL,
                spatial_extent=self._spatial_extent,
                temporal_extent=self._temporal_extent,
                biopar_type=biopar_var
            )

            batch_job: BatchJob = self._execute_datacube(
                biopar_cube,
                output_format='gtiff',
                name=f'biopar_{biopar_var}_{self.tile}',
                job_options=type(self).JOB_OPTIONS_BIOPAR
            )

            output_files = self._download_job_result(
                batch_job,
                var_dir,
                output_format='gtiff',
                name=biopar_var
            )

            self._biopar_results[biopar_var] = \
                type(self)._create_output_dict_gtiff(output_files)

            self._log.info(
                f'Biopar {biopar_var}: '
                f'{len(self._biopar_results[biopar_var])} files downloaded')

        type(self).pickle_write_dict(biopar_pkl,
                                     biopar_dir, self._biopar_results)

        return self._biopar_results

    def _get_initial_output_dict(self) -> dict:
        """
        Create an initial output dictionary with spatial and temporal extents.

        Returns:
            dict: A dictionary with 'info' containing
            'spatial_extent' and 'temporal_extent'.
        """
        output_dict = dict()
        temporal_extent_dt = [datetime.strptime(
            dt, "%Y-%m-%d") for dt in self._temporal_extent]
        output_dict['info'] = {'spatial_extent': self._spatial_extent,
                               'temporal_extent': temporal_extent_dt}
        return output_dict

    def _check_and_load_pickled_dict(self, f_in: Path,
                                     relative_root: Path) -> dict:
        """Check and validate the pickled dictionary.

        Ensures the spatial and temporal extent of the loaded data
        matches the expected values.
        If the file exists and matches the criteria,
        it returns the loaded dictionary.
        Otherwise, it raises appropriate exceptions.

        Args:
            f_in (Dict[str, any]): A dictionary representing
                                    the input file path.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the spatial and/or temporal extent does not match,
            or if 'info' key is not found in the dictionary.

        Returns:
            dict: The loaded dictionary if all checks pass.
        """

        if f_in.is_file():
            dict_in: dict
            with open(f_in, 'rb') as file:
                dict_in = pickle.load(file)

            type(self).set_absolute_paths_dict(relative_root, dict_in)

        else:
            raise FileNotFoundError(f'No such file {f_in}')

        temporal_extent_dt = [datetime.strptime(
            dt, "%Y-%m-%d") for dt in self._temporal_extent]
        if 'info' in dict_in.keys():
            if 'spatial_extent' in dict_in['info'].keys() and \
               'temporal_extent' in dict_in['info'].keys():

                spatial_extent_loaded = dict_in['info']['spatial_extent']
                temporal_extent_loaded = dict_in['info']['temporal_extent']

                # Ensure the spatial and temporal extent
                # of the loaded data matches
                if spatial_extent_loaded == self._spatial_extent and \
                   temporal_extent_loaded == temporal_extent_dt:
                    self._log.info(
                        'Data already downloaded; '
                        'returning results from data.pkl')
                    return dict_in
                else:
                    # Safety to not overwrite already existing data:
                    raise ValueError('Found data.pkl, but spatial and/or'
                                     ' temporal extent does not match. '
                                     'Please remove the provided output '
                                     'directory to continue.')
            else:
                raise ValueError('Could not find "info" in dictionary')

    def _execute_datacube(self, datacube: Union[DataCube, Callable],
                          output_format: Literal['netcdf', 'gtiff'] = 'netcdf',
                          name: str = 'datacube', job_id: str = None,
                          job_options: dict = None) -> BatchJob:
        """
        Executes a batch job using the provided data cube.

        Args:
            datacube (Union[DataCube, Callable]): Either a DataCube instance
                or a callable/partial that returns a DataCube instance.
            output_format (Literal): Select output format
            name (str): Name to display in the openeo process.
            jobid (str): Pick up a certain jobid.
            job_options (dict): A dictionary of job options

        Returns:
            BatchJob: The resulting batchjob after start_and_wait()

        Raises:
            TypeError: If the datacube is not a DataCube instance
                        or a callable.
        """
        if callable(datacube):
            dc = datacube()
            if not isinstance(dc, DataCube):
                raise TypeError(
                    f'Expected a DataCube, got {type(dc).__name__} instead')

        elif isinstance(datacube, DataCube):
            dc = datacube

        else:
            raise TypeError(
                f'Expected a DataCube or callable, '
                f'got {type(datacube).__name__} instead')

        if job_id is None:
            job = dc.create_job(out_format=output_format,
                                title=f'SenETDownload:{name}',
                                job_options=job_options)
            job = job.start_and_wait()
        else:
            job = dc.connection.job(job_id)
            status = job.status()
            if status != 'finished':
                raise NotImplementedError(f'Job status "{status}". '
                                          'No logic implemented '
                                          'to handle this. '
                                          'Please wait until the job has '
                                          '"finished" status.')

        self._log.debug(f'Finished {name}: {job.job_id}')
        return job

    def _download_job_result(self, batch_job: BatchJob,
                             output_path: Path,
                             output_format: Literal['netcdf',
                                                    'gtiff'] = 'netcdf',
                             name: str = 'datacube'):

        if output_format == 'gtiff':
            output_dir = output_path

        elif output_format == 'netcdf':
            output_dir = output_path.parent
        else:
            raise ValueError(f'Format "{output_format}" is not supported')

        output_dir.mkdir(parents=True, exist_ok=True)

        # Not yet optimized for parallel download... Doesn't matter atm
        # It could take a while when the result is available,
        # therefore the retry mechanism:
        retries_left = type(self).DOWNLOAD_RETRIES
        while True:
            try:
                self._log.info(f'Downloading {output_path}')
                job_result = batch_job.get_results()
                job_files = job_result.download_files(output_dir)
                break
            except OpenEoApiError as e:
                if retries_left <= 0:
                    self._log.critical(
                        f'Failed to download. {output_path}. '
                        'Retry count exceeded.')
                    raise e
                self._log.warning(
                    f'Failed to download. {output_path}. '
                    f'Retry in {type(self).RETRY_DELAY} seconds.')
                retries_left = retries_left - 1
                time.sleep(type(self).RETRY_DELAY)

        # Rename logic:
        output_files = []  # Should contain only raster data (no .json!)
        for job_file in job_files:

            if output_format == 'netcdf':
                if job_file.suffix == '.nc':
                    job_file.rename(output_path)
                    output_files.append(output_path)

                elif job_file.name == 'job-results.json':
                    renamed_path = output_path.with_suffix('.json')
                    job_file.rename(renamed_path)

            elif output_format == 'gtiff':
                openeo_date_pattern = r"openEO_\d{4}-\d{2}-\d{2}Z\.tif"
                if job_file.name == 'openEO.tif':
                    renamed_path = job_file.parent / f'{name}.tif'
                    job_file.rename(renamed_path)
                    output_files.append(renamed_path)

                elif re.match(openeo_date_pattern, job_file.name):
                    renamed_path = job_file.parent / \
                        job_file.name.replace('openEO', name)
                    job_file.rename(renamed_path)
                    output_files.append(renamed_path)

                elif job_file.name == 'job-results.json':
                    renamed_path = job_file.parent / f'{name}.json'
                    job_file.rename(renamed_path)

        return output_files

    def _get_datacube_s2(self,
                         eoconn: Optional[Connection] = None) -> DataCube:
        if eoconn is None:
            eoconn = openeo.connect(
                type(self).OPENEO_CDSE_URL).authenticate_oidc()

        cube_properties = {"eo:cloud_cover": lambda val: val <= 95.0}

        s2_bands = eoconn.load_collection(
            self.name_s2,
            temporal_extent=self._temporal_extent,
            spatial_extent=self._spatial_extent,
            bands=type(self).S2_BAND_NAMES,
            properties=cube_properties
        ).resample_spatial(resolution=20)

        # SCL masking
        if self._s2_should_mask:

            # load SCL band
            scl_cube = eoconn.load_collection(
                collection_id="SENTINEL2_L2A",
                bands=["SCL"],
                temporal_extent=self._temporal_extent,
                spatial_extent=self._spatial_extent,
                properties=cube_properties)

            # Compute the SCL dilation mask the same settings as the BIOPAR UDP to be consistent with the biopar variable computation are applied to the S2 bands. 
            scl_dilated_mask = scl_cube.process(
                "to_scl_dilation_mask",
                data=scl_cube,
                scl_band_name="SCL",
                kernel1_size=17,
                kernel2_size=201,
                mask1_values=[2, 4, 5, 6, 7],
                mask2_values=[3, 8, 9, 10, 11],
                erosion_kernel_size=3,
            ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

            s2_bands = s2_bands.mask(scl_dilated_mask)

        # Compute NDVI as well
        s2_bands_ndvi = s2_bands.ndvi(nir='B08', red='B04',
                                      target_band='NDVI')

        # Dekadal compositing
        if self._s2_should_composite:
            s2_bands_ndvi = s2_bands_ndvi.aggregate_temporal_period(
                period='dekad', reducer='median')

        # Interpolation https://processes.openeo.org/#array_interpolate_linear
        if self._s2_should_interpolate:
            s2_bands_ndvi = s2_bands_ndvi.apply_dimension(
                dimension="t", process="array_interpolate_linear"
            )

        return s2_bands_ndvi

    def _get_datacube_s3(self,
                         eoconn: Optional[Connection] = None) -> DataCube:
        if eoconn is None:
            eoconn = openeo.connect(
                type(self).OPENEO_CDSE_URL).authenticate_oidc()

        s3_band1 = eoconn.load_collection(
            self.name_s3,
            temporal_extent=self._temporal_extent,
            spatial_extent=self._spatial_extent,
            bands=type(self).S3_BAND_NAMES1
        )
        
        s3_band2 = eoconn.load_collection(
            self.name_s3,
            temporal_extent=self._temporal_extent,
            spatial_extent=self._spatial_extent,
            bands=type(self).S3_BAND_NAMES2
        )

        s3_band =s3_band1.merge_cubes(s3_band2)
        
        # Further filter the S3 datacube:
        if self._s3_should_filter:
            # 1. Filter only daytime observations
            sza_filter_udf = openeo.UDF.from_file(type(self).UDF_SZA_S3_FILE)
            s3_band = s3_band.apply_dimension(
                process=sza_filter_udf, dimension='t')

            # 2. Select the observation with the fewest clouds
            # when multiple observations are still available
            # for one day.
            cloud_filter_udf = openeo.UDF.from_file(
                type(self).UDF_FILTER_CLOUD_S3_FILE)
            s3_band = s3_band.apply_dimension(
                process=cloud_filter_udf, dimension='t')

        # Cloud masking UDF
        if self._s3_should_mask_clouds:
            cloud_masker_udf = openeo.UDF.from_file(
                type(self).UDF_MASK_CLOUDS_S3_FILE)
            s3_band = s3_band.apply_dimension(
                process=cloud_masker_udf, dimension='t')

        return s3_band

    def _get_datacube_dem(self,
                          eoconn: Optional[Connection] = None) -> DataCube:
        if eoconn is None:
            eoconn = openeo.connect(
                type(self).OPENEO_CDSE_URL).authenticate_oidc()

        dem_band = eoconn.load_collection(
            self.name_dem,
            temporal_extent=["2010-01-01", "2030-12-31"],
            spatial_extent=self._spatial_extent,
            bands=["DEM"],
        )
        dem_band = dem_band.max_time()
        return dem_band

    def _get_datacube_worldcover(self,
                                 eoconn: Optional[Connection] = None
                                 ) -> DataCube:
        """Load the ESA WorldCover land-cover product from OpenEO.

        The collection is chosen automatically based on the temporal extent:
        - start year < 2021  → ``ESA_WORLDCOVER_10M_2020_V1``
        - start year ≥ 2021  → ``ESA_WORLDCOVER_10M_2021_V2``
        """
        if eoconn is None:
            eoconn = openeo.connect(
                type(self).OPENEO_CDSE_URL).authenticate_oidc()

        wc_band = eoconn.load_collection(
            self.name_worldcover,
            temporal_extent=["2000-01-01", "2030-12-31"],
            spatial_extent=self._spatial_extent,
            bands=["MAP"],
        )
        wc_band = wc_band.max_time()
        return wc_band

    def _get_datacube_biopar(self, biopar_var: str,
                             eoconn: Optional[Connection] = None) -> DataCube:
        if eoconn is None:
            eoconn = openeo.connect(
                type(self).OPENEO_CDSE_URL).authenticate_oidc()

        # The biopar UDP handles S2 loading (B03, B04, B08 + angle bands),
        # band selection, and SCL cloud masking internally.
        # Pass spatial_extent, temporal_extent, and biopar_type as
        # top-level parameters — do NOT pass a pre-built data cube.
        biopar_cube = eoconn.datacube_from_process(
            process_id="biopar",
            namespace=type(self).BIOPAR_UDP_URL,
            spatial_extent=self._spatial_extent,
            temporal_extent=self._temporal_extent,
            biopar_type=biopar_var
        )

        # Dekadal compositing (same as S2/NDVI)
        if self._s2_should_composite:
            biopar_cube = biopar_cube.aggregate_temporal_period(
                period='dekad', reducer='median')

        # Linear temporal interpolation to fill gaps (same as S2/NDVI)
        if self._s2_should_interpolate:
            biopar_cube = biopar_cube.apply_dimension(
                dimension="t", process="array_interpolate_linear"
            )

        return biopar_cube

    def _calculate_incidence_angle(self, input_dict: dict) -> dict:
        tmp_dict = input_dict.copy()

        # s2_data = tmp_dict[self.name_s2]
        s3_data = tmp_dict[self.name_s3]
        dem_data = tmp_dict[self.name_dem]
        # s2_dates = list(s2_data['B02'].keys())
        s3_dates = list(s3_data['LST'].keys())

        inc_dict = dict()

        for s3_date in s3_dates:
            # s2_date = find_closest_date(s3_date, s2_dates)

            lowres_lst_file = s3_data['LST'][s3_date]
            # s3_mask_file = s3_data['confidence_in_bitlayers'][s3_date]
            slope_file = dem_data['slo']
            aspect_file = dem_data['asp']
            # altitude_file = dem_data['alt']
            highres_lat_file = s3_data['latHR'][s3_date]
            highres_lon_file = s3_data['lonHR'][s3_date]

            # derive other inputs
            datetime = pd.to_datetime(s3_date)
            doy = datetime.timetuple().tm_yday
            ftime = datetime.hour + (datetime.minute / 60.0)
            # compute incidence angle
            cos_theta_file = lowres_lst_file.parent / \
                lowres_lst_file.name.replace('LST', 'INC')
            inc_dict[s3_date] = cos_theta_file

            self._log.info(f'Calculate incidence angle {s3_date}')
            incidence_angle_tilted(highres_lat_file, highres_lon_file,
                                   doy, ftime,
                                   aspect_file, slope_file,
                                   stdlon=0,
                                   outfile=cos_theta_file)

        return inc_dict

    def _generate_quality_flag(self, input_dict: dict) -> dict:
        """
        This mask grid 'quality_flag' sets good quality pixels to 1.
        """

        tmp_dict = input_dict.copy()
        s3_data = tmp_dict[self.name_s3]
        s3_data['quality_flag'] = dict()
        s3_dates = list(s3_data['LST'].keys())
        quality_flag_dict = dict()

        for s3_date in s3_dates:
            self._log.info(f'Generate quality flag {s3_date}')
            confidence_in: Path = s3_data['confidence_in_bitlayers'][s3_date]
            uncertainty: Path = s3_data['LST_uncertainty'][s3_date]

            f_in_handler: rasterio.io.DatasetReader
            with rasterio.open(confidence_in, 'r') as f_in_handler:
                # Save ref profile for output
                ref_profile: dict = f_in_handler.profile
                # Read band 15 for cloud mask
                s3_cloud_data = f_in_handler.read(15).astype(bool)

            with rasterio.open(uncertainty, 'r') as f_in_handler:
                s3_uncertainty = f_in_handler.read(1)

            # CONDITION1: If no clouds, set values in final data to 1
            final_data = np.logical_not(s3_cloud_data)

            # CONDITION2: If s3 uncertainty is <= 1,
            # set values in final data to 1
            s3_uncertainty = np.nan_to_num(s3_uncertainty, nan=0.0)
            final_data = np.logical_and(final_data, s3_uncertainty <= 1.0)

            # Write data
            f_out = confidence_in.parent / \
                f"quality_flag_{s3_date.strftime('%Y%m%dT%H%M%SZ')}.tif"
            output_profile = ref_profile.copy()
            output_profile['count'] = 1
            output_profile['dtype'] = 'uint8'
            output_profile.pop('nodata')

            f_out_handler: rasterio.io.DatasetWriter
            with rasterio.open(f_out, 'w', **output_profile) as f_out_handler:
                f_out_handler.write(final_data, 1)

            quality_flag_dict[s3_date] = f_out

        return quality_flag_dict

    @staticmethod
    def filter_tifs_by_extent(
            tif_list,
            t_start: datetime,
            t_end: datetime) -> list:
        """Return only those paths whose filename-embedded date falls within
        ``[t_start, t_end]``.  Files with no recognisable date are kept."""
        _pat = re.compile(
            r'.*_(\d{4})-?(\d{2})-?(\d{2})(?:T[^Z]*)?Z?\.tif', re.I)
        filtered = []
        for f in tif_list:
            m = _pat.match(Path(f).name)
            if m:
                fdate = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if t_start <= fdate <= t_end:
                    filtered.append(f)
            else:
                filtered.append(f)  # no date in name → keep
        return filtered

    @staticmethod
    def _find_covering_s3_netcdfs(
            nc_dir: Path,
            t_start: datetime,
            t_end: datetime) -> List[Path]:
        """Return a sorted list of ``datacube_s3_*.nc`` files in *nc_dir*
        whose combined temporal coverage spans ``[t_start, t_end]``
        without internal gaps.

        Filenames are expected to encode their temporal range as
        ``datacube_s3_YYYYMMDD_YYYYMMDD.nc`` (the format produced by
        :attr:`_temporal_pkl_suffix`).  Files that do not match this pattern
        are silently ignored.

        Returns an empty list when the existing files do not fully cover the
        requested extent or when no matching files are found.
        """
        pat = re.compile(r'datacube_s3_(\d{8})_(\d{8})\.nc')
        candidates: List[tuple] = []
        for f in sorted(nc_dir.glob('datacube_s3_*.nc')):
            m = pat.match(f.name)
            if m:
                fs = datetime.strptime(m.group(1), '%Y%m%d')
                fe = datetime.strptime(m.group(2), '%Y%m%d')
                candidates.append((fs, fe, f))
        if not candidates:
            return []
        # Sort by start date and verify contiguous coverage of [t_start, t_end]
        candidates.sort(key=lambda x: x[0])
        if candidates[0][0] > t_start:
            return []  # earliest file starts after the requested start
        covered_up_to = candidates[0][1]
        for fs, fe, _ in candidates[1:]:
            if fs > covered_up_to + timedelta(days=1):
                return []  # gap detected between consecutive files
            covered_up_to = max(covered_up_to, fe)
        if covered_up_to < t_end:
            return []  # files do not reach the requested end date
        # Return only files that overlap the requested range
        return [f for fs, fe, f in candidates if fe >= t_start and fs <= t_end]

    @staticmethod
    def _convert_spatial_extent(
            spatial_extent: Union[Path,
                                  Dict[str, float],
                                  str]) -> Dict[str, float]:
        if isinstance(spatial_extent, dict):
            # dictionary
            return spatial_extent
        elif isinstance(spatial_extent, Path):
            # Shape file of some kind
            if spatial_extent.suffix == '.gpkg':
                # TODO: Check whether this method is working
                aoi = gpd.read_file(spatial_extent)
                aoi = aoi.to_crs(epsg=4326)
                west, south, east, north = aoi.total_bounds
                return {'west': west, 'east': east,
                        'north': north, 'south': south}
        elif isinstance(spatial_extent, str):
            # s2 tile
            import sen_et_openeo.utils
            s2grids_file = Path(
                sen_et_openeo.utils.__file__).parent / 's2grid_bounds.geojson'

            tile = spatial_extent
            gdf = gpd.read_file(s2grids_file)

            tile_data = gdf[gdf['tile'] == tile]
            if not tile_data.empty:
                bounds = tile_data.bounds
                west = bounds['minx'].iloc[0]
                east = bounds['maxx'].iloc[0]
                north = bounds['maxy'].iloc[0]
                south = bounds['miny'].iloc[0]
                return {'west': west, 'east': east,
                        'north': north, 'south': south}
            else:
                raise ValueError(f'{tile} seems not to be a valid S2 tile')

    @staticmethod
    def _add_s2_scale_offset(filelist: List[Path]):
        scale = 0.0001
        offset = 0

        scales = [scale] * 10 + [1]
        offsets = [offset] * 10 + [0]

        for f in filelist:
            with rasterio.open(f, "r+") as handler:
                handler.offsets = offsets
                handler.scales = scales

    @staticmethod
    def _create_output_dict_gtiff(
            filelist: List[Path]) -> Dict[datetime, Path]:

        # Matches both:
        # - openEO date format: SENTINEL2_L2A_2022-01-01Z.tif (dashes, no time)
        # - legacy format: something_20220101T000000Z.tif (no dashes, with time)
        pattern = r".*_(\d{4})-?(\d{2})-?(\d{2})(?:T[^Z]*)?Z\.tif"
        output_dict = dict()
        for f in filelist:
            match = re.match(pattern, f.name)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                parsed_date = datetime(year, month, day)
                output_dict[parsed_date] = f

        return output_dict

    @staticmethod
    def bbox2dict(bbox: rasterio.coords.BoundingBox):
        ret_dict = dict()
        ret_dict['left'] = bbox.left
        ret_dict['right'] = bbox.right
        ret_dict['top'] = bbox.top
        ret_dict['bottom'] = bbox.bottom
        return ret_dict

    @staticmethod
    def _split_geotiff_s2(file_list: List[Path],
                          output_dir: Optional[Path] = None) \
            -> Dict[str, Dict[datetime, Path]]:

        log = SenETDownload.LOGGER
        if output_dir is None:
            output_dir = file_list[0].parent
        else:
            output_dir.mkdir(exist_ok=True, parents=True)

        orig_gtiff_dict = SenETDownload._create_output_dict_gtiff(file_list)
        split_gtiff_dict = dict()

        for f_date, f_in in orig_gtiff_dict.items():

            f_in_handler: rasterio.io.DatasetReader
            with rasterio.open(f_in, 'r') as f_in_handler:
                in_profile = f_in_handler.profile
                bands = in_profile['count']

                out_profile = in_profile.copy()
                out_profile['count'] = 1

                scales = f_in_handler.scales
                offsets = f_in_handler.offsets

                for i in range(0, bands):
                    desc = f_in_handler.descriptions[i]
                    scale = scales[i]
                    offset = offsets[i]
                    data = f_in_handler.read(i + 1)

                    f_out = output_dir / \
                        f'{desc}_{f_date.strftime("%Y%m%dZ")}.tif'

                    # Write file
                    log.debug(f'Write {f_out.name}')
                    f_out_handler: rasterio.io.DatasetWriter
                    nt = SenETDownload.NUM_THREADS_RIO
                    with rasterio.open(f_out, 'w', **out_profile,
                                       num_threads=nt) \
                            as f_out_handler:
                        f_out_handler.write(data, 1)
                        f_out_handler.descriptions = [desc]
                        f_out_handler.scales = [scale]
                        f_out_handler.offsets = [offset]

                    # Add to output dict
                    if desc not in split_gtiff_dict.keys():
                        split_gtiff_dict[desc] = dict()

                    split_gtiff_dict[desc][f_date] = f_out

        return split_gtiff_dict

    @staticmethod
    def _netcdf_to_geotiff_s3(nc_file: Path, output_dir: Optional[Path] = None,
                              force_f32: bool = True) \
            -> Dict[str, Dict[datetime, Path]]:

        if not output_dir:
            output_dir = nc_file.parent
        else:
            output_dir.mkdir(exist_ok=True, parents=True)

        # Convert to geotiff
        gtiff_dict = SenETDownload.netcdf_to_geotiff(nc_file, output_dir)

        # Convert the confidence_in bit values to bit layers,
        # append to gtiff_dict...
        gtiff_dict['confidence_in_bitlayers'] = dict()
        for dt, f in gtiff_dict['confidence_in'].items():
            date_str = f.stem.split('-')[1]
            confidence_bitlayers = f.parent / \
                f"confidence_in_bitlayers_{date_str}.tif"

            if not confidence_bitlayers.exists():
                SenETDownload.convert_s3_confidence_in(f, confidence_bitlayers)

            gtiff_dict['confidence_in_bitlayers'][dt] = confidence_bitlayers

        return gtiff_dict

    @staticmethod
    def _extract_dem_features(dem_file: Path,
                              output_dir: Optional[Path] = None) \
            -> Dict[str, Path]:

        gtiff_dict = dict()

        if output_dir is None:
            output_dir = dem_file.parent
        else:
            output_dir.mkdir(exist_ok=True, parents=True)

        gtiff_dict['alt'] = output_dir / 'DEM_alt.tif'
        gtiff_dict['slo'] = output_dir / 'DEM_slo.tif'
        gtiff_dict['asp'] = output_dir / 'DEM_asp.tif'

        dem_handler: rasterio.io.DatasetReader
        with rasterio.open(dem_file, 'r') as dem_handler:
            alt_data = dem_handler.read(1).astype(np.float32)

            # Extract dem features
            attributes = ['slope_riserun', 'aspect']
            alt_data[alt_data < -10000] = -9999
            rda = rd.rdarray(alt_data, no_data=-9999)
            slo_data, asp_data = [rd.TerrainAttribute(
                rda, attrib=attr) for attr in attributes]
            asp_data = np.deg2rad(asp_data)

            data_dict = dict()
            data_dict['alt'] = alt_data
            data_dict['asp'] = asp_data
            data_dict['slo'] = slo_data

            output_handler: rasterio.io.DatasetWriter
            for feat, f_out in gtiff_dict.items():

                output_profile = dem_handler.profile.copy()
                output_profile['nodata'] = -9999
                output_profile['dtype'] = 'float32'

                with rasterio.open(f_out, 'w',
                                   **output_profile,
                                   num_threads=SenETDownload.NUM_THREADS_RIO) \
                        as output_handler:
                    output_handler.write(data_dict[feat], 1)

        return gtiff_dict

    @staticmethod
    def netcdf_to_geotiff(input_file: Path,
                          output_dir: Path, write_dims=True,
                          force_f32: bool = True) \
            -> Dict[str, Dict[datetime, Path]]:
        """
        Convert a NetCDF file to GeoTIFF format.

        Args:
            input_file (Path): Path to the input NetCDF file.
            output_dir (Path): Directory to save the output GeoTIFF files.
            write_dims (Optional, bool): Write dimension 'x'/'lon' and
                'y'/'lat' to GeoTIFF files.
            force_f32 (Optional, bool): Force conversion to float32
                if data type is float64.

        Returns:
            dict: Dictionary with variable names as keys and sub-dictionaries
                with time steps as keys and corresponding GeoTIFF
                file paths as values.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            xar_in = rioxarray.open_rasterio(input_file)
        except KeyError:
            xar_in = xr.open_dataset(input_file)

        epsg = None
        if xar_in.rio.crs is None:
            # Assume 'EPSG:4326'
            epsg = 'EPSG:4326'
            xar_in.rio.write_crs(epsg, inplace=True)
        else:
            epsg = xar_in.rio.crs.to_epsg()
            if epsg is None:
                # Properly set EPSG.
                if xar_in.rio.crs.to_dict()['datum'] == 'WGS84':
                    epsg = 'EPSG:4326'
                    xar_in.rio.write_crs(epsg, inplace=True)
            else:
                SenETDownload.LOGGER.warning('Could not find proper EPSG')

        output_dict = dict()
        data_arrays = {var: xar_in[var] for var in xar_in.data_vars}

        for name, data_array in data_arrays.items():
            output_dict[name] = dict()
            if 't' in data_array.dims:
                for t in xar_in['t'].values:
                    t = pd.Timestamp(t)
                    str_date = t.strftime('%Y%m%dT%H%M%SZ')
                    output_file = output_dir / f"{name}-{str_date}.tif"
                    dt = datetime(t.year, t.month, t.day, t.hour,
                                  t.minute, t.second, t.microsecond)
                    output_dict[name][dt] = output_file
                    if output_file.exists():
                        continue  # already converted in a prior run
                    data = data_array.sel(t=t)
                    if data.dtype == 'float64' and force_f32:
                        data = data.astype('float32')
                    data.rio.write_crs(epsg, inplace=True)
                    data.rio.to_raster(output_file, compress='deflate')

        # Write dimension grid
        if write_dims:
            tmp_x = output_dir / 'x_tmp.tif'
            tmp_y = output_dir / 'y_tmp.tif'
            SenETDownload.write_dim_grids(
                xar_in, tmp_x, tmp_y, force_f32=force_f32)

            name_x = 'x'
            name_y = 'y'
            if xar_in.rio.crs.to_epsg() in [4326]:
                name_x = 'lon'
                name_y = 'lat'

            output_dict[name_x] = dict()
            output_dict[name_y] = dict()

            for t in xar_in['t'].values:
                t = pd.Timestamp(t)
                str_date = t.strftime('%Y%m%dT%H%M%SZ')

                output_file_x = output_dir / f"{name_x}-{str_date}.tif"
                output_file_y = output_dir / f"{name_y}-{str_date}.tif"

                dt = datetime(t.year, t.month, t.day, t.hour,
                              t.minute, t.second, t.microsecond)

                output_dict[name_x][dt] = output_file_x
                output_dict[name_y][dt] = output_file_y

                if output_file_x.exists() and output_file_y.exists():
                    continue  # already written in a prior run
                shutil.copyfile(tmp_x, output_file_x)
                shutil.copyfile(tmp_y, output_file_y)

            tmp_x.unlink()
            tmp_y.unlink()

        return output_dict

    @staticmethod
    def write_dim_grids(xar_in: xr.DataArray,
                        output_x: Path, output_y: Path,
                        force_f32: bool = True):
        """
        Writes x and y dimension grids of a DataArray to raster files.

        Extracts 'x'/'lon' and 'y'/'lat' dimensions, creates expanded grids,
        and writes them to specified raster files.

        Args:
            xar_in (xr.DataArray): Input DataArray with spatial dimensions.
            output_x (Path): File path for the x-dimension raster output.
            output_y (Path): File path for the y-dimension raster output.
            force_f32 (Optional, bool): Force conversion to float32
                if data type is float64.
        """
        x = y = None
        crs = xar_in.rio.crs
        if 'x' in xar_in.dims:
            x = xar_in['x'].values
        elif 'lon' in xar_in.dims:
            x = xar_in['lon'].values
        else:
            raise ValueError('Could not find proper x dimension in data')

        if 'y' in xar_in.dims:
            y = xar_in['y'].values
        elif 'lat' in xar_in.dims:
            y = xar_in['y'].values
        else:
            raise ValueError('Could not find proper y dimension in data')

        if x.dtype == np.float64 and force_f32:
            x = x.astype(np.float32)

        if y.dtype == np.float64 and force_f32:
            y = y.astype(np.float32)

        if x is not None and y is not None:
            x_expanded, y_expanded = np.meshgrid(x, y)
            x_out = xr.DataArray(
                x_expanded,
                dims=["y", "x"],
                coords={
                    "y": y,
                    "x": x
                }
            )
            x_out.rio.write_crs(crs, inplace=True)
            x_out.rio.to_raster(output_x, compress='deflate')

            y_out = xr.DataArray(
                y_expanded,
                dims=["y", "x"],
                coords={
                    "y": y,
                    "x": x
                }
            )
            y_out.rio.write_crs(crs, inplace=True)
            y_out.rio.to_raster(output_y, compress='deflate')

    @staticmethod
    def convert_s3_confidence_in(input_file: Path, output_file: Path):
        """
        Convert Sentinel-3 confidence layer to separate bit layers.

        Args:
            input_file (Path): Path to the input raster file.
            output_file (Path): Path to the output raster file.

        """
        no_of_bits = 16
        bit_text_code = {0: 'coastline',
                         1: 'ocean',
                         2: 'tidal',
                         3: 'land',
                         4: 'inland_water',
                         5: 'unfilled',
                         6: 'spare',
                         7: 'spare',
                         8: 'cosmetic',
                         9: 'duplicate',
                         10: 'day',
                         11: 'twilight',
                         12: 'sun_glint',
                         13: 'snow',
                         14: 'summary_cloud',
                         15: 'summary_pointing'
                         }

        with rasterio.open(input_file, 'r') as src:
            src_profile = src.profile

            dst_profile = src_profile.copy()
            dst_profile['dtype'] = 'uint8'
            dst_profile['nodata'] = 255
            dst_profile['count'] = no_of_bits

            nt = SenETDownload.NUM_THREADS_RIO
            with rasterio.open(output_file, 'w',
                               **dst_profile,
                               num_threads=nt) as dst:
                # Convert to uint8 for proper bitwise operations
                # (currently float32)
                in_data = src.read(1)
                mask = np.isnan(in_data)
                in_data_uint16 = in_data.astype(np.uint16)

                for i in range(0, no_of_bits):
                    binary_grid = (in_data_uint16 % 2).astype(np.bool_)
                    binary_grid[mask] = 255

                    in_data_uint16 = np.right_shift(in_data_uint16, 1)
                    dst.set_band_description(i + 1, bit_text_code[i])
                    dst.write(binary_grid, i + 1)

    @staticmethod
    def reference_warp_gdal(src_ds: Path, ref_ds: Path,
                            dst_ds: Optional[Path] = None,
                            resampling: str = 'near',
                            res: Union[str, tuple] = 'to_ref') -> None:
        """
        Warp a raster dataset to match the spatial reference, extent,
        and resolution of a reference dataset.

        We prefer this method because of gdalwarps robustness rather
        than the rasterio warp.

        Args:
            src_ds (Path): Path to the source raster dataset to be warped.
            ref_ds (Path): Path to the reference raster dataset used to define
                the output spatial reference, extent and resolution.
            dst_ds (Path, optional): Path to save the warped raster dataset.
                If none, the src_ds will be overwritten.
            resampling (str, optional): Resampling method to use during
                warping. Defaults to 'near'.
            res (str, tuple, optional):
                - 'auto': Let gdalwarp decide the resolution.
                - 'to_ref': Resample to the reference resolution.
                - tuple/list: A tuple containing the x and y resolution values.

        Returns:
            None
        """
        if dst_ds is not None:
            tmp_dst_ds = dst_ds
        else:
            tmp_dst_ds = src_ds.parent / f'tmp_{src_ds.name}'

        ref_ds = Path(ref_ds)
        if not ref_ds.exists():
            raise FileNotFoundError(
                f'Reference dataset does not exist: {ref_ds}')

        try:
            with rasterio.open(ref_ds, 'r') as ref:
                ref_transform = ref.transform
                # ref_res = ref_transform[0]
                ref_crs = ref.crs
                ref_bounds = ref.bounds
        except rasterio.errors.RasterioIOError as ex:
            raise RuntimeError(
                f'Reference dataset is unreadable or not in a supported '
                f'format: {ref_ds}. Original error: {ex}') from ex

        dst_output_bounds = (ref_bounds.left, ref_bounds.bottom,
                             ref_bounds.right, ref_bounds.top)
        if res == 'auto':
            dst_x_res = None
            dst_y_res = None
        elif res == 'to_ref':
            dst_x_res = ref_transform.a
            dst_y_res = ref_transform.e
        elif isinstance(res, (tuple, list)):
            dst_x_res = res[0]
            dst_y_res = res[1]
        else:
            raise ValueError(f'{res} not recognised')

        dst_srs = ref_crs.to_wkt()  #adjusted to properly set the SRS in the output file was dst_srs = ref_crs['init']

        creation_opts = ('COMPRESS=DEFLATE', 'TILED=YES',
                         'BLOCKXSIZE=256', 'BLOCKYSIZE=256')

        warp_options = gdal.WarpOptions(outputBounds=dst_output_bounds,
                                        xRes=dst_x_res,
                                        yRes=dst_y_res,
                                        resampleAlg=resampling,
                                        dstSRS=dst_srs,
                                        creationOptions=creation_opts
                                        )
        # Set performance tweaks:
        gdal.SetConfigOption('GDAL_NUM_THREADS',
                             SenETDownload.NUM_THREADS_GDAL)
        gdal.SetConfigOption('GDAL_CACHEMAX', '2048')

        # Do warp:
        gdal.Warp(str(tmp_dst_ds), str(src_ds),
                  format='GTiff', options=warp_options)

        # Unset performance tweaks:
        gdal.SetConfigOption('GDAL_NUM_THREADS', None)
        gdal.SetConfigOption('GDAL_CACHEMAX', None)

        if dst_ds is None:
            src_ds.unlink()
            tmp_dst_ds.rename(src_ds)

    @staticmethod
    def remove_key_and_file(input_dict: dict,
                            key_to_remove, pop=True) -> dict:
        """Removes a key from a dictionary and deletes the corresponding file
            if the value is a Path.

         This method recursively removes the specified key from the dictionary.
         If the value associated with the key is a dictionary,
            it recursively removes all keys in the nested dictionary.

        Args:
            input_dict (dict): The dictionary from which to remove the key.
            key_to_remove: The key to remove from the dictionary.
            pop: Whether to remove the key from the dictionary.
                If False, only the files will be removed.
        """
        log = SenETDownload.LOGGER
        if key_to_remove not in input_dict.keys():
            raise ValueError(f'Key {key_to_remove} not found.')

        value = input_dict[key_to_remove]
        if isinstance(value, dict):
            for key in value.keys():
                SenETDownload.remove_key_and_file(value, key, pop=False)
        elif isinstance(value, Path) or issubclass(value, Path):
            log.debug(f'Removing {value}')
            value.unlink()
        else:
            # Skip
            pass

        if pop:
            input_dict.pop(key_to_remove)

        return input_dict

    @staticmethod
    def set_relative_paths_dict(relative_root: Path, input_dictionary: dict):
        for key, value in input_dictionary.items():
            if isinstance(value, dict):
                SenETDownload.set_relative_paths_dict(relative_root, value)

            elif isinstance(value, Path):
                input_dictionary[key] = Path(
                    os.path.relpath(value, relative_root))

            elif isinstance(value, list):
                if all(isinstance(item, Path) for item in value):
                    for i in range(0, len(value)):
                        value[i] = Path(
                            os.path.relpath(value[i], relative_root))

            else:
                pass
                # Skip

    @staticmethod
    def _iter_paths(d: dict):
        """Recursively yield all Path values stored in a (nested) dict."""
        for v in d.values():
            if isinstance(v, Path):
                yield v
            elif isinstance(v, dict):
                yield from SenETDownload._iter_paths(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Path):
                        yield item

    @staticmethod
    def set_absolute_paths_dict(relative_root: Path, input_dictionary: dict):
        for key, value in input_dictionary.items():
            if isinstance(value, dict):
                SenETDownload.set_absolute_paths_dict(relative_root, value)

            elif isinstance(value, Path):
                input_dictionary[key] = (relative_root / value).resolve()

            elif isinstance(value, list):
                if all(isinstance(item, Path) for item in value):
                    for i in range(0, len(value)):
                        value[i] = (relative_root / value[i]).resolve()

            else:
                pass
                # Skip

    @staticmethod
    def pickle_write_dict(output_file: Path,
                          relative_root: Path, input_dictionary: dict):
        tmp_dict = deepcopy(input_dictionary)

        SenETDownload.set_relative_paths_dict(relative_root, tmp_dict)

        with open(output_file, 'wb') as pkl_file:
            pickle.dump(tmp_dict, pkl_file)

    @staticmethod
    def reference_warp_rio(src_ds: Path, ref_ds: Path, dst_ds: Optional[Path],
                           resampling: int = Resampling.nearest) -> None:
        """
        This method utilizes rasterio's warp functionality to warp
        the source raster dataset to match the spatial reference, extent
        and resolution of the reference raster dataset.

        Args:
            src_ds (Path): Path to the source raster dataset to be warped.
            ref_ds (Path): Path to the reference raster dataset used to define
                the output spatial reference, extent and resolution.
            dst_ds (Path, optional): Path to save the warped raster dataset.
                If None, src_ds will be overwritten
            resampling (int, optional): Resampling method to use
                during warping. Defaults to Resampling.nearest.

        Returns:
            None

        """
        if dst_ds is not None:
            tmp_dst_ds = dst_ds
        else:
            tmp_dst_ds = src_ds.with_stem(f'tmp_{src_ds.stem}')

        with rasterio.open(src_ds, 'r') as src:
            src_profile = src.profile
            src_transform = src.transform
            src_crs = src.crs
            src_width = src.width
            src_height = src.height
            src_bounds = src.bounds
            src_bound_dict = SenETDownload.bbox2dict(src_bounds)
            src_nodata = src.nodata

        with rasterio.open(ref_ds, 'r') as ref:
            ref_transform = ref.transform
            ref_res = ref_transform[0]
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height

        dst_transform, dst_width, \
            dst_height = rasterio.warp.calculate_default_transform(
                src_crs=src_crs,
                dst_crs=ref_crs,
                width=src_width,
                height=src_height,
                dst_width=ref_width,
                dst_height=ref_height,
                **src_bound_dict
            )
        # Set a and e to the ref's a and e.
        # Otherwise, it will slightly deviate.
        dst_transform = rasterio.Affine(ref_transform.a, dst_transform.b,
                                        dst_transform.c, dst_transform.d,
                                        ref_transform.e, dst_transform.f)

        dst_profile = src_profile.copy()
        dst_profile['crs'] = ref_crs
        dst_profile['transform'] = dst_transform
        dst_profile['width'] = dst_width
        dst_profile['height'] = dst_height

        with rasterio.open(src_ds, 'r') as src, \
            rasterio.open(tmp_dst_ds, 'w',
                          **dst_profile,
                          num_threads=SenETDownload.NUM_THREADS_RIO) as dst:
            for i in src.indexes:
                src_data = src.read(i)
                dst_data, dst_transform = rasterio.warp.reproject(
                    src_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transfrom=dst_transform,
                    dst_crs=ref_crs,
                    dst_resolution=ref_res,
                    resampling=resampling,
                    num_threads=4
                )
                dst_data = dst_data[0][:][:]
                dst.write(dst_data, indexes=i)
                dst.update_tags(i, **src.tags(i))
            dst.update_tags(**src.tags())

        if dst_ds is None:
            src_ds.unlink()
            tmp_dst_ds.rename(src_ds)

    @staticmethod
    def warp_s3_to_S2(infile: Path, outfile: Path,
                      template: Path, nodatavalue):
        """
        Warps an input file to match the spatial properties of a template file.

        Args:
            infile (Path): Path to the input file to be warped.
            outfile (Path): Path where the warped output file will be saved.
            template (Path): Path to the template file that provides
                the desired spatial properties.
            nodatavalue (float): Value to use for NoData in the output file.
        """

        # retrieve properties from template
        r = gdal.Open(str(template))
        proj = r.GetProjection()
        gt = r.GetGeoTransform()
        sizeX = r.RasterXSize
        sizeY = r.RasterYSize
        extent = [gt[0], gt[3]+gt[5]*sizeY, gt[0]+gt[1]*sizeX, gt[3]]

        creation_opts = ('COMPRESS=DEFLATE', 'TILED=YES',
                         'BLOCKXSIZE=256', 'BLOCKYSIZE=256')

        # Set performance tweaks:
        gdal.SetConfigOption('GDAL_NUM_THREADS',
                             SenETDownload.NUM_THREADS_GDAL)
        gdal.SetConfigOption('GDAL_CACHEMAX', '2048')

        gdal.Warp(str(outfile),
                  str(infile),
                  format="GTiff",
                  dstSRS=proj,
                  xRes=gt[1],
                  yRes=gt[5],
                  outputBounds=extent,
                  resampleAlg='cubicspline',
                  dstNodata=nodatavalue,
                  srcNodata=nodatavalue,
                  creationOptions=creation_opts)

        # Unset performance tweaks:
        gdal.SetConfigOption('GDAL_NUM_THREADS', None)
        gdal.SetConfigOption('GDAL_CACHEMAX', None)

    @staticmethod
    def get_scaling_data():
        def normalize(input_dict):
            tmp_dict = dict()
            if input_dict['scale'] is not None:
                tmp_dict['scale'] = 1.0 / float(input_dict['scale'])
            else:
                tmp_dict['scale'] = None

            tmp_dict['dtype'] = input_dict['dtype'].__name__

            tmp_dict['nodata'] = input_dict['nodata']

            if 'offset' in tmp_dict.keys():
                tmp_dict['offset'] = input_dict['offset']
            else:
                tmp_dict['offset'] = 0.0

            return tmp_dict

        # Normalize data
        scaling_data_orig = OUTPUT_SCALING.copy()
        for key in scaling_data_orig:
            scaling_data_orig[key] = normalize(scaling_data_orig[key])

        output_dict = dict()

        # S2 L2A
        sen2_dict = dict()

        for band in SenETDownload.S2_BAND_NAMES:
            sen2_dict[band] = scaling_data_orig['S2-refl']

        output_dict['SENTINEL2_L2A'] = sen2_dict

        # S3 LST
        sen3_dict = dict()
        sen3_dict['LST'] = scaling_data_orig['S3-LST']
        sen3_dict['LST_uncertainty'] = scaling_data_orig['S3-LST']
        sen3_dict['exception'] = None
        sen3_dict['confidence_in'] = {
            'dtype': 'uint16', 'scale': None, 'nodata': None, 'offset': 0}
        sen3_dict['sunAzimuthAngles'] = scaling_data_orig['S3-geo']
        sen3_dict['sunZenithAngles'] = scaling_data_orig['S3-geo']
        sen3_dict['viewAzimuthAngles'] = scaling_data_orig['S3-geo']
        sen3_dict['viewZenithAngles'] = scaling_data_orig['S3-geo']
        sen3_dict['confidence_in_bitlayers'] = None
        sen3_dict['lat'] = scaling_data_orig['S3-geo']
        sen3_dict['lon'] = scaling_data_orig['S3-geo']
        sen3_dict['sunAzimuthAnglesHR'] = scaling_data_orig['S3-geo']
        sen3_dict['sunZenithAnglesHR'] = scaling_data_orig['S3-geo']
        sen3_dict['viewAzimuthAnglesHR'] = scaling_data_orig['S3-geo']
        sen3_dict['viewZenithAnglesHR'] = scaling_data_orig['S3-geo']
        sen3_dict['latHR'] = scaling_data_orig['S3-geo']
        sen3_dict['lonHR'] = scaling_data_orig['S3-geo']
        sen3_dict['inc'] = {'scale': None,
                            'dtype': 'float32', 'nodata': -9999, 'offset': 0}
        sen3_dict['quality_flag'] = None

        output_dict['SENTINEL3_SLSTR_L2_LST'] = sen3_dict

        dem_dict = dict()
        dem_dict['alt'] = scaling_data_orig['DEM-alt-20m']
        dem_dict['slo'] = scaling_data_orig['DEM-slo-20m']
        dem_dict['asp'] = scaling_data_orig['DEM-asp-20m']
        output_dict['COPERNICUS_30'] = dem_dict

        return output_dict

    @staticmethod
    def to_scaled_raster(src_ds: Path,
                         scale: float = 1.0,
                         offset: float = 0.0,
                         dtype: str = 'uint16',
                         nodata: Union[int, float] = None,
                         dst_ds: Path = None):
        """
        Scales and offsets raster data from a source dataset,
        and saves it to a destination dataset.

        Parameters:
        src_ds (Path): Path to the source dataset.
        scale (Optional[float]): Scale factor for the data (default is 1.0).
        offset (float): Offset value to be subtracted from the data
                        (default is 0.0).
        dtype (str): Desired data type of the output dataset
                    (default is 'uint16').
        nodata: Value to use for 'nodata' in the output dataset
                (default is None).
        dst_ds (Optional[Path]): Path to the destination dataset.
        If None, the source dataset is modified in-place (default is None).

        Returns:
        None
        """
        intermediary_dtype = np.float32

        if dst_ds is not None:
            tmp_dst_ds = dst_ds
        else:
            tmp_dst_ds = src_ds.parent / \
                f'tmp_to_scaled_raster_{src_ds.stem}.tif'

        dtype_np = np.dtype(dtype)

        if offset is None:
            offset = 0.0

        src_handler: rasterio.io.DatasetReader
        with rasterio.open(src_ds, 'r') as src_handler:
            src_profile = src_handler.profile
            dst_profile = src_profile.copy()

            dst_profile['dtype'] = dtype
            dst_profile['nodata'] = nodata

            # Force lossless compression
            if dst_profile['compress'].lower() not in ('deflate', 'lzw'):
                dst_profile['compress'] = 'deflate'

            dst_handler: rasterio.io.DatasetWriter
            with rasterio.open(tmp_dst_ds, 'w', **dst_profile,
                               num_threads=SenETDownload.NUM_THREADS_RIO) \
                    as dst_handler:

                # Set scales if applicable
                if scale is not None:
                    dst_handler.offsets = [offset] * src_handler.count
                    dst_handler.scales = [scale] * src_handler.count

                for band in range(src_handler.count):
                    # Read data
                    in_data: np.ndarray = src_handler.read(
                        band + 1).astype(intermediary_dtype)

                    # Scale if applicable
                    if scale is not None:
                        out_data = (in_data - offset) / scale
                    else:
                        out_data = in_data

                    # Create and apply nodata mask
                    in_nodata = src_handler.nodata
                    if nodata is not None and in_nodata is not None:
                        in_nodata = src_handler.nodata
                        if np.isnan(in_nodata):
                            nodata_mask = np.isnan(in_data)
                        else:
                            nodata_mask = in_data == in_nodata

                        out_data[nodata_mask] = nodata

                    # convert to dtype and write
                    out_data = out_data.astype(dtype_np)
                    dst_handler.write(out_data, band + 1)

        if dst_ds is None:
            src_ds.unlink()
            tmp_dst_ds.rename(src_ds)

    @staticmethod
    def get_current_memory_usage():
        """
        from:
        https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_info
            - rss: aka “Resident Set Size”, this is the non-swapped physical
                memory a process has used.
                On UNIX it matches “top“‘s RES column).
                On Windows this is an alias for wset field and it matches
                “Mem Usage” column of taskmgr.exe.
            - vms: aka “Virtual Memory Size”, this is the total amount of
                virtual memory used by the process.
                On UNIX it matches “top“‘s VIRT column.
                On Windows this is an alias for pagefile field and it
                matches “Mem Usage” “VM Size” column of taskmgr.exe.
            - shared: (Linux) memory that could be potentially shared with
                other processes. This matches “top“‘s SHR column).
            - text (Linux, BSD): aka TRS (text resident set) the amount of
                memory devoted to executable code.
                This matches “top“‘s CODE column).
            - data (Linux, BSD): aka DRS (data resident set) the amount of
                physical memory devoted to other than executable code.
                It matches “top“‘s DATA column).
            - lib (Linux): the memory used by
                shared libraries.
            - dirty (Linux): the number of dirty pages.
            - pfaults (macOS): number of page faults.
            - pageins (macOS): number of actual pageins.

        """

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info


class BlockHelper:
    def __init__(self, lo_res_file: Path,
                 hi_res_file: Path,
                 low_res_x_block_size: int = 35,
                 low_res_y_block_size: int = 35,
                 tmp_dir: Path = Path('/tmp/blockhelper'),
                 cleanup: bool = True):
        self._lo_res_file = lo_res_file
        self._hi_res_file = hi_res_file
        self._lo_res_x_block_size = low_res_x_block_size
        self._lo_res_y_block_size = low_res_y_block_size
        self._tmp_dir = tmp_dir
        self._cleanup = cleanup
        self._gdal_theads = 'ALL_CPUS'

        # To be set in _initialize:
        self._hi_res_x_block_size: Optional[int] = None
        self._hi_res_y_block_size: Optional[int] = None

        # Store already processed files here:
        self._hi_res_file_dict: Dict[int, Path] = dict()
        self._lo_res_file_dict: Dict[int, Path] = dict()

        self._initialize()

    def _initialize(self):
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        lores_handler: rasterio.io.DatasetReader
        hires_handler: rasterio.io.DatasetReader
        with rasterio.open(self._lo_res_file, 'r') as lores_handler, \
                rasterio.open(self._hi_res_file, 'r') as hires_handler:

            # Calculate the hi res block size
            # hi_res_x_block_size = hires_width / low_res_x_windows
            # hi_res_y_block_size = hires_height / low_res_y_windows
            self._hi_res_x_block_size = int(np.ceil(
                self._lo_res_x_block_size * 1098 / 20))
            self._hi_res_y_block_size = int(np.ceil(
                self._lo_res_y_block_size * 1098 / 20))

            self.lores_windows = type(self).generate_windows(
                lores_handler,
                self._lo_res_x_block_size,
                self._lo_res_y_block_size)

            self.hires_windows = type(self).generate_windows(
                hires_handler,
                self._hi_res_x_block_size,
                self._hi_res_y_block_size)

            # Sanity check:
            if len(self.lores_windows) != len(self.hires_windows):
                raise ValueError(
                    'hires_windows and lores_windows are not equal in length')

            self.window_count = len(self.lores_windows)

    def get_lo_res_file(self, window_no: int):
        if window_no >= self.window_count:
            raise ValueError(f'Invalid window number: {window_no}. '
                             f'It must be less than the total number'
                             f' of windows: {self.window_count}.')

        if window_no in self._lo_res_file_dict.keys():
            return self._lo_res_file_dict[window_no]

        _, window = self.lores_windows[window_no]
        f_out = self._tmp_dir / \
            f'{self._lo_res_file.stem}'
        f'_LORES_WINDOW{window_no}'
        f'{self._lo_res_file.suffix}'

        self._get_window_file(self._lo_res_file, f_out, window)

        self._lo_res_file_dict[window_no] = f_out
        return f_out

    def get_hi_res_file(self, window_no: int):
        if window_no >= self.window_count:
            raise ValueError(f'Invalid window number: {window_no}. '
                             f'It must be less than the total number'
                             f' of windows: {self.window_count}.')

        if window_no in self._hi_res_file_dict.keys():
            return self._hi_res_file_dict[window_no]

        _, window = self.hires_windows[window_no]

        f_out = self._tmp_dir / \
            f'{self._hi_res_file.stem}'
        f'_HIRES_WINDOW{window_no}'
        f'{self._hi_res_file.suffix}'

        self._get_window_file(self._hi_res_file, f_out, window)

        self._hi_res_file_dict[window_no] = f_out
        return f_out

    def get_hi_res_files(self):
        f_out_list = list()
        for i in self.window_count:
            f_out = self.get_hi_res_file(i)
            f_out_list.append(f_out)

        return f_out_list

    def get_lo_res_files(self):
        f_out_list = list()
        for i in self.window_count:
            f_out = self.get_lo_res_file(i)
            f_out_list.append(f_out)

        return f_out_list

    def get_hi_lo_window(self, window_no: int):
        _, window = self.lores_windows[window_no]
        return window

    def get_hi_res_window(self, window_no: int):
        _, window = self.hires_windows[window_no]
        return window

    def _get_window_file(self, f_in: Path, f_out: Path, window):
        options = gdal.TranslateOptions(srcWin=(window.col_off,
                                                window.row_off,
                                                window.width,
                                                window.height),
                                        creationOptions=('COMPRESS=DEFLATE',
                                                         'TILED=NO',
                                                         'INTERLEAVE=BAND')
                                        )

        # Set performance_tweaks
        gdal.SetConfigOption('GDAL_NUM_THREADS', self._gdal_theads)
        gdal.SetConfigOption('GDAL_CACHEMAX', '2048')

        gdal.Translate(str(f_out), str(f_in), options=options)

        # Unset performance tweaks:
        gdal.SetConfigOption('GDAL_NUM_THREADS', None)
        gdal.SetConfigOption('GDAL_CACHEMAX', None)
        return f_out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cleanup:
            for f in self._hi_res_file_dict.values():
                if f.exists():
                    f.unlink()

            for f in self._lo_res_file_dict.values():
                if f.exists():
                    f.unlink()

        return False

    @staticmethod
    def generate_windows(ds, width=256, height=256):
        """
        Function to generate rasterio windows according to
        the rasterio dataset.
        """
        y_size = ds.height
        x_size = ds.width
        rows = int(math.ceil(float(y_size) / float(height)))
        cols = int(math.ceil(float(x_size) / float(width)))

        windows = []
        for row in range(0, rows):
            for col in range(0, cols):
                xCoor = width * col
                yCoor = height * row
                x_plus = width
                y_plus = height
                if x_size <= xCoor + x_plus:
                    x_plus = x_size - xCoor
                if y_size <= yCoor + y_plus:
                    y_plus = y_size - yCoor

                ji = (row, col)
                window = rasterio.windows.Window(
                    width=x_plus, height=y_plus, col_off=xCoor, row_off=yCoor)
                windows.append((ji, window))

        return windows
