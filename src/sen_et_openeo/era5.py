import os
import glob
from loguru import logger
import pandas as pd
from pathlib import Path
import numpy as np
import netCDF4
import datetime
import copy

from satio.timeseries import Timeseries
from satio.collections import DiskCollection

from sen_et_openeo.utils.geoloader import (_getECMWFTempInterpData,
                                           _getECMWFIntegratedData)
from sen_et_openeo.utils.warping import warp_in_memory
from sen_et_openeo.utils.timedate import _bracketing_dates
from sen_et_openeo.utils.meteo import (comp_air_temp_inputs,
                                       comp_air_temp)
from sen_et_openeo.ts import TimeSeriesProcessor, _TimeSeriesTimer

ERA5_BANDS_DICT = {25000: ['t2m', 'z', 'd2m', 'sp', 'v100',
                           'u100', 'ssrdc', 'ssrd']}
ERA5_BANDS_DICT_DOWNLOAD = ["2m_temperature", "z",
                            "2m_dewpoint_temperature",
                            "surface_pressure",
                            "100m_v_component_of_wind",
                            "100m_u_component_of_wind",
                            "surface_solar_radiation_downward_clear_sky",
                            "surface_solar_radiation_downwards"]


def get_default_rsi_meta():
    return {
        'ERA5':
            {
                "air_temperature": {
                    'bands': ['t2m', 'd2m', 'z', 'sp'],
                    'native_res': 25000,
                    'func': comp_air_temp}
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
        s = {}

        s["variable"] = variables
        s["product_type"] = "reanalysis"
        s["date"] = date_start+"/"+date_end
        s["time"] = [str(t).zfill(2)+":00" for t in range(0, 24, 1)]
        if area is not None:
            s["area"] = area
        s["format"] = "netcdf"

        # Connect to the server and download the data
        c = cdsapi.Client()
        c.retrieve("reanalysis-era5-single-levels", s, downloadpath)


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
        self.timer = _TimeSeriesTimer(25000)

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
            nctime = ncfile.variables['time']
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
