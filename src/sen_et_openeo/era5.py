import os
import glob
import re
import zipfile
from collections import defaultdict
from loguru import logger
import pandas as pd
from pathlib import Path
import numpy as np
import netCDF4
import datetime
import copy
from osgeo import gdal

from sen_et_openeo.utils.timeseries import Timeseries
from sen_et_openeo.utils.collections import DiskCollection

from sen_et_openeo.utils.geoloader import (_getECMWFTempInterpData,
                                           _getECMWFIntegratedData)
from sen_et_openeo.utils.warping import warp_in_memory
from sen_et_openeo.utils.timedate import _bracketing_dates
from sen_et_openeo.utils.meteo import (comp_air_temp_inputs,
                                       comp_air_temp)
from sen_et_openeo.ts import TimeSeriesProcessor

ERA5_BANDS_DICT = {25000: ['t2m', 'z', 'd2m', 'sp',
                           'v100', 'u100', 'ssrd'
                           ]}
ERA5_BANDS_DICT_DOWNLOAD = ["2m_temperature",
                            "geopotential",
                            "2m_dewpoint_temperature",
                            "surface_pressure",
                            "100m_v_component_of_wind",
                            "100m_u_component_of_wind",
                            "surface_solar_radiation_downwards"
                            ]

# Mapping from GRIB_ELEMENT values (WMO codes, case-insensitive) to internal
# ERA5 variable names used throughout this codebase.
# Variables whose GRIB_ELEMENT already matches the internal name (Z, SP, SSRD)
# are not listed here.
_GRIB_ELEMENT_TO_INTERNAL = {
    '2t':   't2m',
    '2d':   'd2m',
    'var246 of table 228 of center ecmwf': 'u100',
    'var247 of table 228 of center ecmwf': 'v100',
}



def _get_grib_element(band_meta: dict) -> str:
    """Return the GRIB_ELEMENT (WMO code) from a GDAL GRIB band metadata dict.
    
    Only uses GRIB_ELEMENT for identification; GRIB_SHORT_NAME is considered
    unreliable and is not consulted.
    """
    return str(band_meta.get('GRIB_ELEMENT', ''))


def _parse_valid_time_epoch(meta: dict) -> int:
    """Parse GRIB_VALID_TIME metadata into a Unix epoch integer, or -1."""
    raw = meta.get('GRIB_VALID_TIME') or meta.get('valid_time') or ''
    m = re.search(r'(\d{9,})', str(raw))
    return int(m.group(1)) if m else -1


def _convert_grib_to_netcdf(grib_path: Path, nc_path: Path) -> None:
    """Convert a GRIB file to a CF-compliant NetCDF4 file.

    Variable short names extracted from GRIB metadata (via GRIB_ELEMENT WMO code)
    are remapped to the internal ERA5 naming convention via
    ``_GRIB_ELEMENT_TO_INTERNAL`` (e.g. '2t' -> 't2m', '100u' -> 'u100').
    Names not listed in the mapping are kept as-is.
    The time axis is built from GRIB_VALID_TIME epoch values found in the band
    metadata so that the resulting file is directly compatible with
    ``ERA5TimeSeriesProcessor.load_data()``.
    """
    ds = gdal.Open(str(grib_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Could not open GRIB file: {grib_path}')

    gt = ds.GetGeoTransform()
    x_size, y_size = ds.RasterXSize, ds.RasterYSize
    lon = gt[0] + (np.arange(x_size, dtype=np.float64) + 0.5) * gt[1]
    lat = gt[3] + (np.arange(y_size, dtype=np.float64) + 0.5) * gt[5]

    # Collect band records grouped by internal variable name.
    # Each entry: list of (valid_time_epoch, band_index, metadata_dict)
    by_var = defaultdict(list)
    all_times = set()

    for idx in range(1, ds.RasterCount + 1):
        b = ds.GetRasterBand(idx)
        meta = b.GetMetadata() or {}
        grib_elem = _get_grib_element(meta).lower()
        if not grib_elem:
            logger.warning(
                f'Band {idx} in {grib_path} has no GRIB_ELEMENT metadata; '
                'skipping band')
            continue
        # Remap GRIB_ELEMENT to internal name via case-insensitive lookup
        internal = _GRIB_ELEMENT_TO_INTERNAL.get(grib_elem, grib_elem)
        valid_time = _parse_valid_time_epoch(meta)
        by_var[internal].append((valid_time, idx, meta))
        if valid_time >= 0:
            all_times.add(valid_time)

    if not all_times:
        raise RuntimeError(
            f'No GRIB_VALID_TIME metadata found in {grib_path}; '
            'cannot construct NetCDF time axis')

    times = sorted(all_times)
    time_to_idx = {t: i for i, t in enumerate(times)}

    nc_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(nc_path), 'w', format='NETCDF4') as nc:
        nc.createDimension('valid_time', len(times))
        nc.createDimension('latitude', y_size)
        nc.createDimension('longitude', x_size)

        tvar = nc.createVariable('valid_time', 'i8', ('valid_time',))
        tvar[:] = np.array(times, dtype=np.int64)
        tvar.units = 'seconds since 1970-01-01 00:00:00'
        tvar.calendar = 'proleptic_gregorian'
        tvar.standard_name = 'time'
        tvar.long_name = 'time'

        yvar = nc.createVariable('latitude', 'f8', ('latitude',))
        yvar[:] = lat
        yvar.units = 'degrees_north'
        yvar.standard_name = 'latitude'

        xvar = nc.createVariable('longitude', 'f8', ('longitude',))
        xvar[:] = lon
        xvar.units = 'degrees_east'
        xvar.standard_name = 'longitude'

        for internal_name, records in sorted(by_var.items()):
            var = nc.createVariable(
                internal_name, 'f4',
                ('valid_time', 'latitude', 'longitude'),
                zlib=True, complevel=4,
                fill_value=np.float32(np.nan),
            )
            data = np.full(
                (len(times), y_size, x_size), np.nan, dtype=np.float32)
            sample_units = None
            sample_long_name = None

            for (vt, band_idx, meta) in records:
                if vt < 0:
                    continue
                b = ds.GetRasterBand(band_idx)
                arr = b.ReadAsArray().astype(np.float32)
                nodata = b.GetNoDataValue()
                scale = b.GetScale() if b.GetScale() is not None else 1.0
                offset = b.GetOffset() if b.GetOffset() is not None else 0.0
                if nodata is not None and not np.isnan(nodata):
                    arr[arr == nodata] = np.nan
                arr = arr * scale + offset
                data[time_to_idx[vt]] = arr
                if sample_units is None:
                    sample_units = (
                        meta.get('GRIB_UNIT') or meta.get('units'))
                if sample_long_name is None:
                    sample_long_name = (
                        meta.get('GRIB_COMMENT') or meta.get('long_name'))

            var[:] = data
            if sample_units is not None:
                var.units = sample_units
            if sample_long_name is not None:
                var.long_name = sample_long_name

    ds = None


def comp_wind_speed(u100, v100):
    """Compute wind speed (m/s) from 100m u and v wind components."""
    return np.sqrt(u100**2 + v100**2)


def comp_vapour_pressure(d2m):
    """Compute vapour pressure (mb/hPa) from 2m dewpoint temperature (K).
    Uses the Magnus formula.
    """
    Td_C = d2m - 273.15
    return 6.1078 * np.exp(17.269 * Td_C / (235.5 + Td_C))


def comp_air_pressure_mb(sp):
    """Convert surface pressure from Pa to mb."""
    return sp / 100.0


def get_default_rsi_meta():
    return {
        'ERA5':
            {
                "air_temperature": {
                    'bands': ['t2m', 'd2m', 'z', 'sp'],
                    'native_res': 25000,
                    'func': comp_air_temp},
                "vapour_pressure": {
                    'bands': ['d2m'],
                    'native_res': 25000,
                    'func': comp_vapour_pressure},
                "air_pressure": {
                    'bands': ['sp'],
                    'native_res': 25000,
                    'func': comp_air_pressure_mb},
                "wind_speed": {
                    'bands': ['u100', 'v100'],
                    'native_res': 25000,
                    'func': comp_wind_speed},
            }
    }


def get_era5(date_start, date_end, downloadpath, area=None,
             variables=ERA5_BANDS_DICT_DOWNLOAD):
    """
    area should be defined as a list of bounds in latlon:
    [North, West, South, East]
    """
    if not os.path.exists(downloadpath):

        import cdsapi
        s = {
            "variable": variables,
            "product_type": "reanalysis",
            "date": date_start + "/" + date_end,
            "time": [str(t).zfill(2) + ":00" for t in range(0, 24, 1)],
            # GRIB is currently the most robust ERA5 output format.
            "data_format": "grib",
            "download_format": "unarchived"
        }
        if area is not None:
            s["area"] = area

        # Connect to the server and download the data
        c = cdsapi.Client()

        c.retrieve("reanalysis-era5-single-levels", s, downloadpath)

        # Normalize ZIP responses (if any) to a single asset in downloadpath.
        with open(downloadpath, 'rb') as f:
            magic = f.read(4)

        if magic.startswith(b'PK'):
            with zipfile.ZipFile(downloadpath, 'r') as zf:
                nc_members = [m for m in zf.namelist()
                              if m.lower().endswith('.nc')]
                grib_members = [m for m in zf.namelist()
                                if m.lower().endswith('.grib')
                                or m.lower().endswith('.grb')]
                members = nc_members if len(nc_members) > 0 else grib_members
                if len(members) == 0:
                    raise RuntimeError(
                        f'CDS returned ZIP but no .nc/.grib file was found: '
                        f'{downloadpath}')
                with zf.open(members[0], 'r') as src, \
                        open(downloadpath, 'wb') as dst:
                    dst.write(src.read())

            with open(downloadpath, 'rb') as f:
                magic = f.read(4)

        # Convert GRIB to NetCDF using the metadata-aware converter that
        # preserves ERA5 variable names and builds a proper time axis.
        if magic.startswith(b'GRIB'):
            grib_path = Path(downloadpath).with_suffix('.grib')
            os.replace(downloadpath, str(grib_path))
            logger.info(
                'CDS returned GRIB; converting to NetCDF with '
                'metadata-aware conversion: {}', grib_path)
            _convert_grib_to_netcdf(grib_path, Path(downloadpath))

        # Fail fast if retrieval succeeded but did not return a valid NetCDF file.
        try:
            nc = netCDF4.Dataset(downloadpath, 'r')
            nc.close()
        except Exception as e:
            kind = 'unknown'
            with open(downloadpath, 'rb') as f:
                head = f.read(64)
            if head.startswith(b'{') or head.startswith(b'['):
                kind = 'json'
            elif head.lower().startswith(b'<!doctype') or head.lower().startswith(b'<html'):
                kind = 'html'
            elif head.startswith(b'GRIB'):
                kind = 'grib'
            raise RuntimeError(
                f'Invalid NetCDF output at {downloadpath}. '
                f'Detected payload kind: {kind}. '
                'Verify CDS credentials/terms and file conversion.') from e


class ERA5Collection(DiskCollection):

    sensor = 'ERA5'

    @classmethod
    def from_folders(cls, folder, s2grid=None):
        df = cls.build_products_df(folder)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    @property
    def supported_resolutions(self):
        return [1000]

    @ property
    def supported_bands(self):
        return ERA5_BANDS_DICT.get(25000)

    def get_band_filenames():
        pass

    @ classmethod
    def build_products_df(cls, folder):

        products = []
        tiles = [Path(x).stem for x in glob.glob(str(Path(folder) / '*'))]
        for tile in tiles:
            products.extend(glob.glob(
                str(Path(folder) / tile / '*.nc')))

        entries = [cls.era5_entry(f) for f in products]
        if len(entries):
            df = pd.DataFrame(entries)
        else:
            df = pd.DataFrame([], columns=['date',
                                           'tile',
                                           'path',
                                           'level'])
        return df

    @ staticmethod
    def era5_entry(filename):
        """
        """
        date = Path(filename).stem
        tile = Path(filename).parent.stem
        entry = dict(date=pd.to_datetime(date),
                     tile=tile,
                     level='',
                     path=filename)

        return entry

    def filter_dates(self, dates):
        df = self.df.copy()
        df['day'] = df.date.dt.strftime('%Y-%m-%d')
        df = df[df.day.isin(dates)]
        start_date = df.sort_values('date').iloc[0].date
        end_date = df.sort_values('date').iloc[-1].date
        return self._clone(df=df, start_date=start_date, end_date=end_date)

    def check_tiledates(self, tile, timestamps,
                        outdir):
        """
        Check the collection to get certain dates
        of a certain Sentinel-2 tile.
        In case the products you are looking for, are not
        yet available in the collection, they are downloaded
        from CDS here!
        """
        # check which dates are needed
        # use buffer of + and - 1 day
        first = timestamps[0] - pd.Timedelta(days=1)
        last = timestamps[-1] + pd.Timedelta(days=1)
        dates = [x.strftime("%Y-%m-%d") for x in timestamps]
        dates = list(dict.fromkeys(dates))
        dates.insert(0, first.strftime("%Y-%m-%d"))
        dates.append(last.strftime("%Y-%m-%d"))

        # check available products for the tile:
        df = self.df[self.df.tile.isin([tile])]
        logger.info(f'{len(df)} meteo products found for '
                    f'tile {tile}')
        avail_dates = list(df.date.values)
        avail_dates = [np.datetime_as_string(x, unit='D') for x in avail_dates]

        get_dates = [d for d in dates if d not in avail_dates]

        if len(get_dates) > 0:

            self.prepare_products(outdir, tile,
                                  get_dates)

        # re-generate collection
        collection = ERA5Collection.from_path(outdir)

        return collection

    def prepare_products(self, outdir, tile,
                         dates):

        bounds = self._s2grid.loc[self._s2grid['tile']
                                  == tile].bounds.values[0]
        # apply 0.2° buffer to make sure full extent is covered!
        area = [bounds[3] + 0.2, bounds[0] - 0.2,
                bounds[1] - 0.2, bounds[2] + 0.2]

        # download the data
        def _process_date(date):
            target = Path(outdir) / tile / f'{date}.nc'
            target.parent.mkdir(exist_ok=True, parents=True)
            get_era5(date, date, str(target), area=area)

        # loop over all required dates
        for date in dates:
            _process_date(date)


class ERA5TimeSeriesProcessor(TimeSeriesProcessor):

    def __init__(self, timestamps, elev, demfile, time_zone,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timestamps = timestamps
        self.elev = elev
        self.demfile = demfile
        self.time_zone = time_zone

    @ property
    def _reflectance(self):
        return False

    @ property
    def supported_bands(self):
        return ERA5_BANDS_DICT

    @ property
    def supported_rsis(self):
        if self._supported_rsis is None:
            rsis_dict = {}
            rsi_res = {r: self._rsi_meta[r]['native_res']
                       for r in self._rsi_meta.keys()}
            rsis_dict[25000] = [v for v, r in rsi_res.items() if r == 25000]
            self._supported_rsis = rsis_dict

        return self._supported_rsis

    def load_data(self, dtype=np.float32):

        timeseries = []

        # derive all bands which are required
        bands = copy.deepcopy(self.settings.get('bands', []))
        rsis = self.settings.get('rsis', [])
        if len(rsis) > 0:
            for rsi in rsis:
                bands.extend(self._rsi_meta[rsi].get('bands'))
            # get rid of duplicates
            bands = list(set(bands))

        if len(bands) == 0:
            raise ValueError('No ERA5 bands to load!')

        for t in self.timestamps:
            # get the right files and times to read
            datestr = t.date().strftime(format='%Y-%m-%d')
            if datestr not in self.collection.df.day.values:
                logger.error(
                    f'No ERA5 data available for date {datestr},'
                    ' skipping!')
                timeseries.append(None)
                continue
            ecmwf_data_file = self.collection.df.loc[
                self.collection.df.day == datestr].path.values[0]
            ncfile = netCDF4.Dataset(ecmwf_data_file, 'r')
            # Find the location of bracketing dates
            if 'time' in ncfile.variables:
                nctime = ncfile.variables['time']
            elif 'valid_time' in ncfile.variables:
                nctime = ncfile.variables['valid_time']
            else:
                raise ValueError('No time variable found in ERA5 file!')
            nctimes = netCDF4.num2date(
                nctime[:], nctime.units, nctime.calendar)
            beforeI, afterI, frac = _bracketing_dates(nctimes, t)

            # read individual bands
            data = []
            for b in bands:
                if b == 'ssrd':
                    date_local = (t + datetime.timedelta(
                        hours=self.time_zone)).date()
                    midnight_local = datetime.datetime.combine(
                        date_local, datetime.time())
                    midnight_UTC = midnight_local - datetime.timedelta(
                        hours=self.time_zone)
                    d, gt, proj = _getECMWFIntegratedData(ecmwf_data_file,
                                                          b, midnight_UTC,
                                                          time_window=24)
                else:
                    d, gt, proj = _getECMWFTempInterpData(ecmwf_data_file,
                                                          b, beforeI,
                                                          afterI, frac)
                data.append(d)

            # stack all bands
            timeseries.append(np.stack(data, axis=0))
            ncfile.close()

        # stack all data to build timeseries object
        valid = [self.timestamps[i] for i, v in enumerate(timeseries)
                 if v is not None]
        timeseries = [ts for ts in timeseries if ts is not None]
        timeseries = np.stack(timeseries, axis=1)
        timeseries = timeseries.astype(dtype)

        attrs = {'sensor': 'ERA5'}

        ts = Timeseries(timeseries, valid, bands, attrs)

        return ts, gt, proj

    def compute_ts(self):

        ts, gt, proj = self.load_data()

        rsis = self.settings.get('rsis', [])
        bands = self.settings.get('bands', [])

        if len(rsis) > 0:
            if 'air_temperature' in rsis:
                # first compute vapour pressure and air pressure
                to_compute = ['vapour_pressure', 'air_pressure']
                rsidata = ts.compute_rsis(*to_compute,
                                          rsi_meta=self._rsi_meta,
                                          bands_scaling=1)
                # merge with ts
                ts = ts.merge(rsidata)
                # now compute inputs for air temperature
                rsidata = comp_air_temp_inputs(ts)
                # merge with ts
                ts = ts.merge(rsidata)

                rsis_left = [x for x in rsis if x not in
                             ['vapour_pressure', 'air_pressure',
                              'air_temperature']]
            else:
                rsis_left = rsis.copy()

            if len(rsis_left) > 0:
                # now compute the others
                rsidata = ts.compute_rsis(*rsis_left,
                                          rsi_meta=self._rsi_meta,
                                          bands_scaling=1)
                # merge with ts
                ts = ts.merge(rsidata)

            # resample all timeseries
            resampled = []
            for var in range(ts.data.shape[0]):
                resampled.append(warp_in_memory(np.squeeze(ts.data[var, ...]),
                                                gt, proj, self.demfile))
            resampled = np.stack(resampled, axis=0)
            ts_res = Timeseries(resampled, ts.timestamps,
                                ts.bands, ts.attrs)

            if 'air_temperature' in rsis:
                tair = comp_air_temp(ts_res, self.elev)
                ts_res = ts_res.merge(tair)

            # only select the ones that are requested in settings
            requested = copy.deepcopy(rsis)
            if len(bands) > 0:
                requested.extend(bands)
            ts_res = ts_res.select_bands(requested)

        else:
            logger.info('Resampling meteo bands...')

            # resample all timeseries
            resampled = []
            for var in range(ts.data.shape[0]):
                resampled.append(warp_in_memory(np.squeeze(ts.data[var, ...]),
                                                gt, proj, self.demfile))
            resampled = np.stack(resampled, axis=0)
            ts_res = Timeseries(resampled, ts.timestamps,
                                ts.bands, ts.attrs)

        return ts_res
