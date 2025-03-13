import datetime
import joblib
from typing import Dict, List, Callable
import dateutil.parser

import pandas as pd
import numpy as np
from loguru import logger


def load_timeseries(collection, *bands,
                    mask_and_scale=False, **kwargs) -> 'Timeseries':
    """
    Returns a Timeseries from a filtered collection
    """
    bands = list(bands)

    if mask_and_scale:
        data = collection.load(bands=bands,
                               mask_and_scale=mask_and_scale,
                               **kwargs)
    else:
        # Disk Collection doesn't have mask_and_scale kwarg, avoid raising
        data = collection.load(bands=bands,
                               **kwargs)

    arr = [data[b].values for b in bands]

    if not all([d.shape == arr[0].shape for d in arr]):
        raise ValueError(f"Bands {bands} do not have all the same "
                         "resolution.")

    timestamp = data[bands[0]].timestamp.values
    timestamp = np.array([dateutil.parser.parse(str(s)) for s in timestamp])

    arr = np.array(arr)

    # if arr.dtype == np.int16:
    #     logger.warning("Data is int16, forcing uint16")
    #     arr[arr < 0] = 0
    #     arr = arr.astype(np.uint16)

    attrs = {'sensor': collection.sensor}

    return Timeseries(arr, timestamp, bands, attrs)


class Timeseries:

    from sen_et_openeo.utils.features import Features

    def __init__(self,
                 data: np.ndarray,
                 timestamps: List,
                 bands: List,
                 attrs: dict = dict()):
        if data.ndim != 4:
            raise ValueError("Invalid data input shape")
        if data.shape[0] != len(bands):
            raise ValueError(
                "Inconsistent data dimension 0 and number of bands")
        if data.shape[1] != len(timestamps):
            raise ValueError(
                "Inconsistent data dimension 1 and number of timestamps")

        self.data = data
        self.timestamps = np.array(timestamps)
        self.bands = list(bands)
        self.attrs = attrs

    def _clone(self, data=None, timestamps=None,
               bands=None, attrs: dict = None) -> 'Timeseries':
        data = self.data if data is None else data
        timestamps = self.timestamps if timestamps is None else timestamps
        bands = self.bands if bands is None else bands
        attrs = self.attrs if attrs is None else attrs

        return Timeseries(data, timestamps, bands, attrs)

    def composite(self,
                  mode='median',
                  freq=7,
                  window=None,
                  start='20190101',
                  end='20200101') -> 'Timeseries':
        """
        Return a composite timeseries for a specific mode. The `freq`, `before`
        `after` parameters are in days.
        """
        from sen_et_openeo.utils.composite import calculate_moving_composite

        if isinstance(mode, dict):
            if not self.bands[0] in mode.keys():
                raise ValueError((f'{self.bands[0]} not in '
                                  'composite mode settings.'))
            else:
                mode = mode[self.bands[0]]

        new_arr, new_timestamps = calculate_moving_composite(
            self.data,
            self.timestamps,
            mode=mode,
            start=start,
            end=end,
            freq=freq,
            window=window)

        return self._clone(data=new_arr, timestamps=new_timestamps)  # type: ignore  # NOQA

    def interpolate(self, inplace=False) -> 'Timeseries':
        """
        Interpolate missing values (np.nan for float32 arrays or 0 for uint16)
        Other dtypes are not supported.
        """
        from sen_et_openeo.utils.interpolation import interpolate_4d

        if inplace:
            data = self.data
        else:
            data = self.data.copy()

        """
        there is a bug in the interpolate function. For a full timeseries
        of 0 the first pixel can be different from zero after the interpolation
        this monkey patch ensures that all zeros go out as all zeros
        """

        if data.dtype == np.uint16:
            pre_interp_mask = (data[0] > 0).sum(axis=0) > 0
        elif data.dtype == np.float32:
            # in case there are inf values, convert to nan
            data[np.isinf(data)] = np.nan

            pre_interp_mask = (~np.isnan(data[0])).sum(axis=0) > 0
        else:
            raise TypeError(f"dtype {data.dtype} not supported.")

        new_data = interpolate_4d(data)

        new_data[:, :, ~pre_interp_mask] = (0 if data.dtype == np.uint16
                                            else np.nan)

        return self._clone(data=new_data)

    def mask(self, mask, nodata_value=None, drop_nodata=True) -> 'Timeseries':
        """
        Mask data values with the given 2d or 3d mask.
        Values False or 0 in the mask will be set to nodata_value.
        Timestamps with no valid data will be dropped by the Timeseries.
        """
        if nodata_value is None:
            if self.data.dtype == np.uint16:
                nodata_value = 0
            elif self.data.dtype == np.float32:
                nodata_value = np.nan
            else:
                raise ValueError(f'dtype {self.data.dtype} not supported.')

        if ((self.data.dtype == np.uint16) &
                ((nodata_value < 0) | (nodata_value >= 2**16))):
            raise ValueError("`nodata_value` for uint16 timeseries "
                             "should be between 0 and 65535")

        mask = np.broadcast_to(mask, self.data.shape)

        masked_data = self.data.copy()
        masked_data[~mask] = nodata_value

        new = self._clone(data=masked_data)

        if drop_nodata:
            new = new.drop_nodata()

        return new

    def drop_nodata(self, ref_band=0):
        """
        Return new timeseries without nodata timestamps from ref_band
        """
        if self.data.dtype == np.uint16:
            nodata_value = 0
            s = self.data[ref_band].sum(axis=(1, 2))
            nodata_ids = np.where(s == nodata_value)[0]
        elif self.data.dtype == np.float32:
            # Check for all-nan timestamps
            s = np.isnan(self.data[ref_band]).sum(axis=(1, 2))
            nodata_ids = np.where(s == np.prod(
                self.data[ref_band].shape[1:3]))[0]
        else:
            raise ValueError(f'dtype {self.data.dtype} not supported.')

        return self.drop_ids(nodata_ids)

    def drop_ids(self, ids) -> 'Timeseries':
        """
        Return new timeseries withouth the given timestamps
        ids.
        """
        valid_ids = [i for i in range(self.data.shape[1])
                     if i not in ids]
        new_data = self.data[:, valid_ids, :, :]
        new_timestamps = self.timestamps[valid_ids]

        return self._clone(data=new_data, timestamps=new_timestamps)

    def select_bands(self, bands) -> 'Timeseries':
        """
        Returns timeseries from the selected bands
        """
        bands_ids = []
        for b in bands:
            if b not in self.bands:
                raise ValueError(f"Band {b} not available.")
        bands_ids = [self.bands.index(b) for b in bands]

        new_data = self.data[bands_ids, ...]

        return self._clone(data=new_data, bands=bands)

    def _select_closest_timestamp_idx(self,
                                      dt: datetime,
                                      tolerance_hrs=24,
                                      allow_missing=False):

        closest_idx = np.argmin(
            abs(np.array(list(map(lambda x: x.timestamp() - dt.timestamp(),
                                  self.timestamps)))))

        closest = self.timestamps[closest_idx]
        td = closest - dt

        if abs(td.total_seconds() / 3600) > tolerance_hrs:

            if allow_missing:
                closest_idx = None

            else:
                raise ValueError("No timestamps available at"
                                 f" less than {tolerance_hrs} hrs. Closest to"
                                 f" {dt} is {closest} at"
                                 f" {td.seconds / 3600} hrs.")

        return closest_idx

    def select_timestamps(self,
                          timestamps: List[datetime.datetime],
                          allow_missing=False,
                          tolerance_hrs=24,
                          resample=False):

        if type(timestamps) is datetime.datetime:
            timestamps = [timestamps]

        idx = [self._select_closest_timestamp_idx(dt,
                                                  tolerance_hrs,
                                                  allow_missing)
               for dt in timestamps]

        idx = [i for i in idx if i is not None]

        if len(idx) == 0:
            raise ValueError(f"No matching timestamps in the two Timeseries")

        new_data = self.data[:, idx, ...]

        if resample:
            newtimestamps = timestamps
        else:
            newtimestamps = list(np.array(self.timestamps)[idx])

        return self._clone(data=new_data,
                           timestamps=newtimestamps)

    def combine_bands(self,
                      func: Callable,
                      bands: List,
                      output_bands: List,
                      bands_scaling: float = 10000) -> 'Timeseries':

        new_ts = self.select_bands(bands)

        if bands_scaling is None:
            bands_scaling = 1

        old_dtype = new_ts.data.dtype

        if (new_ts.data.dtype == np.uint16) & (bands_scaling == 1):
            logger.warning("Combining bands of dtype uint16 "
                           "without scaling values to reflectance.")

        # convert to float before computing rsis
        new_ts.data = new_ts.data.astype(np.float32) / bands_scaling

        if old_dtype == np.uint16:
            # If we came originally from dtype uint16
            # then nodata values (0) should be explicitly replaced
            # by NaN
            new_ts.data[new_ts.data == 0] = np.nan

        with np.errstate(divide='ignore', invalid='ignore'):
            # selecting axis 0 of data as it will be singleton for a
            # single band. Ignore division for 0 or nans warnings.
            new_data = func(*[new_ts[b].data[0] for b in bands])

        if new_data.ndim == 3:
            new_data = np.expand_dims(new_data, axis=0)

        return self._clone(data=new_data, bands=output_bands)

    def merge(self, *others, fill_missing_timestamps=False) -> 'Timeseries':
        """
        Returns a new instance merging data and names of current feature with
        multiple timeseries.
        """
        feat = self
        for ot in others:
            feat = feat._merge(
                ot, fill_missing_timestamps=fill_missing_timestamps)

        return feat

    def _merge(self, other, fill_missing_timestamps=False) -> 'Timeseries':
        """
        Merge two timeseries with same spatial
        extension. If the same band is present in `other` and `self`
        the one in `other` will be discarded
        If `fill_missing_timestamps` is set to True, the result will be the
        combined timestamps of both original timeseries and missing
        acquisitions in one or the other will be filled with nodata.
        If `fill_missing_timestamps` is set to False, the timestamps
        of both timeseries need to be identical
        """
        import xarray as xr

        if (self.data.shape[-2:] != other.data.shape[-2:]):
            raise ValueError("Cannot merge two timeseries with "
                             "different canvas shape")

        new_other_bands = [b for b in other.bands if b not in self.bands]

        if len(new_other_bands) == 0:
            # No new bands
            return self

        if not fill_missing_timestamps:
            # Check for identical timestamps
            if any(self.timestamps != other.timestamps):
                raise ValueError("Cannot merge two timeseries "
                                 "with different time vectors when "
                                 "`fill_missing_timestamps` is False.")

        # Convert original and new data to xr.DataArray for
        # convenient concatenation
        orig_da = self.to_xarray()
        new_other_da = other.select_bands(new_other_bands).to_xarray()

        if orig_da.dtype == np.uint16:
            # Fill missing timestamps with no data value 0
            merged_da = xr.concat([orig_da, new_other_da], dim='bands',
                                  join='outer', fill_value=0)
        elif orig_da.dtype == np.float32:
            # Fill missing timestamps with NaN
            merged_da = xr.concat([orig_da, new_other_da], dim='bands',
                                  join='outer')
        else:
            raise TypeError(f"dtype {orig_da.dtype} not supported.")

        new_timestamps = np.array([pd.Timestamp(x).to_pydatetime()
                                   for x in merged_da.time.values])

        return self._clone(data=merged_da.data,
                           bands=list(merged_da.bands.values),
                           timestamps=new_timestamps)

    def merge_time(self, *others) -> 'Timeseries':
        """
        Returns a new instance merging data and names of current timeseries
        with multiple timeseries.
        """

        ts = self
        for ot in others:
            ts = ts._merge_time(ot)

        return ts

    def _merge_time(self, other) -> 'Timeseries':
        """Returns a new instance merging multiple time series
        of the same bands and same spatial extension across
        the time axis.

        Returns:
            Timeseries: time-merged Timeseries
        """

        import xarray as xr

        if (self.data.shape[-2:] != other.data.shape[-2:]):
            raise ValueError("Cannot merge two timeseries with "
                             "different canvas shape")

        if self.bands != other.bands:
            raise ValueError("Cannot time-merge two timeseries with "
                             "different bands")

        overlapping_timestamps = [t for t in other.timestamps
                                  if t in self.timestamps]

        if len(overlapping_timestamps) > 0:
            raise NotImplementedError(('Cannot time-merge two Timeseries '
                                       'with overlapping timestamps'))

        # Convert original and new data to xr.DataArray for
        # convenient concatenation
        orig_da = self.to_xarray()
        new_other_da = other.to_xarray()

        # Join both arrays on the time dimension
        merged_da = xr.concat([orig_da, new_other_da], dim='time',
                              join='outer')

        # Get the merged timestamps
        new_timestamps = np.array([pd.Timestamp(x).to_pydatetime()
                                   for x in merged_da.time.values])

        # Back to Timeseries
        return self._clone(data=merged_da.data,
                           bands=list(merged_da.bands.values),
                           timestamps=new_timestamps)

    def downsample(self) -> 'Timeseries':
        """
        Downsamples timeseries to half resolution in the space dimensions.
        Only works for uint16. For other types, see `imresize` method.
        """
        from sen_et_openeo.utils.resample import block_downsample_band3d

        new_data = np.array([block_downsample_band3d(self[b].data[0])
                             for b in self.bands])
        return self._clone(data=new_data)

    def upsample(self) -> 'Timeseries':
        """
        Upsamples timeseries to double the resolution in the space dimensions.
        Using a nearest neighbor method.
        For other resampling methods/scalings, see `imresize` method.
        """
        from sen_et_openeo.utils.resample import upsample

        new_data = np.array([upsample(self[b].data[0])
                             for b in self.bands])
        return self._clone(data=new_data)

    def imresize(self, scaling=1, order=1,
                 shape=None, anti_aliasing=True):
        """
        Resample the timeseries spatially. When upsampling uses
        `skimage.transform.rescale` with the given order (method) given.
        When downsampling either uses `block_resize` when possible or
        `rescale` if the pixels dimensions require it.
        """
        from sen_et_openeo.utils.resample import imresize

        new_data = []
        for b in self.bands:
            band_data = self[b].data[0]
            new_band = []
            for i in range(band_data.shape[0]):
                new_band.append(imresize(band_data[i], scaling, order,
                                         shape, anti_aliasing))

            new_data.append(new_band)

        new_data = np.array(new_data)

        return self._clone(data=new_data)

    def resample_time(self, periods: int) -> 'Timeseries':
        """
        Resample timeseries along the time dimensions using
        `scipy.signal.resample`.
        Timestamps should be regularly spaced as this method
        uses an FFT transform.
        """
        import scipy

        new_data = scipy.signal.resample(self.data, periods, axis=1)
        new_timestamps = pd.date_range(start=self.timestamps[0],
                                       end=self.timestamps[-1],
                                       periods=periods).to_pydatetime()

        return self._clone(data=new_data, timestamps=new_timestamps)

    def compute_band_features(self,
                              func: Callable,
                              band: str,
                              features_names: List,
                              chunk_size: int = None,
                              **kwargs):
        """
        `func` should be a function that acts on a numpy array of
        shape [time, y, x] along the time dimension, returning an array of
        shape [len(features_names), y, x].

        The function is called as `func(arr, **kwargs)`
        """
        # data = np.squeeze(self.select_bands([band]).data)
        from sen_et_openeo.utils.features import Features, compute_chunked_features

        data = self.select_bands([band]).data[0]

        if chunk_size is None:
            features = func(data, **kwargs)
        else:
            n_features = len(features_names)
            features = compute_chunked_features(data,
                                                func,
                                                n_features,
                                                chunk_size=chunk_size,
                                                **kwargs)

        return Features(features, features_names)

    def features_from_dict(self,
                           resolution,
                           features_meta=None,
                           chunk_size=None):
        from sen_et_openeo.utils.features import Features

        if features_meta is None:
            from sen_et_openeo.utils.features import FEATURES_META
            features_meta = FEATURES_META

        features = []
        for k, v in features_meta.items():
            f = v['function']
            params = v.get('parameters', {})
            feature_bands = v.get('bands', self.bands)
            bands = [b for b in feature_bands if b in self.bands]
            for b in bands:
                features_names = [
                    f'{b}-{vn}-{resolution}m' for vn in v['names']]
                features.append(
                    self.compute_band_features(f, b, features_names,
                                               chunk_size=chunk_size,
                                               **params))

        if len(features) > 1:
            features = Features.from_features(*features)
        elif len(features) == 1:
            features = features[0]

        if type(features) is list:
            features = None

        return features

    def compute_rsis(self,
                     *rsi_names,
                     rsi_meta: Dict = None,
                     bands_scaling: int = 10000,
                     interpolate_inf=True) -> 'Timeseries':
        """
        Returns a Timeseries from the combination of bands.

        `bands_scaling` is set to 10000 by default to scale S2 bands to
        reflectance
        """
        from sen_et_openeo.utils.rsindices import RSI_META

        rsi_meta = rsi_meta if rsi_meta is not None else {}
        rsi_meta = {**RSI_META.get(self.attrs['sensor'], {}), **rsi_meta}

        rsis_ts = None

        for rsi_name in rsi_names:
            ts = self._compute_rsi(rsi_name,
                                   rsi_meta=rsi_meta,
                                   bands_scaling=bands_scaling,
                                   interpolate_inf=interpolate_inf)

            if rsis_ts is None:
                rsis_ts = ts
            else:
                rsis_ts = rsis_ts.merge(ts)

        return rsis_ts

    def _compute_rsi(self,
                     rsi_name: str,
                     rsi_meta: Dict = None,
                     bands_scaling: int = 10000,
                     interpolate_inf=True):
        """
        Returns a Timeseries from the combination of bands
        """
        from sen_et_openeo.utils.rsindices import get_rsi_function

        meta = rsi_meta[rsi_name]

        func = get_rsi_function(rsi_name, meta=meta)

        rsi_bands = meta['bands']

        output_bands = meta.get('output_bands', [rsi_name])

        rsi_ts = self.combine_bands(func,
                                    rsi_bands,
                                    output_bands=output_bands,
                                    bands_scaling=bands_scaling)

        clamp = meta.get('clamp_range', False)
        range_min, range_max = meta.get('range', (None, None))

        with np.errstate(invalid='ignore'):
            if clamp:
                rsi_ts.data[rsi_ts.data < range_min] = range_min
                rsi_ts.data[rsi_ts.data > range_max] = range_max
            else:
                if range_min is not None:
                    rsi_ts.data[rsi_ts.data < range_min] = np.nan
                if range_max is not None:
                    rsi_ts.data[rsi_ts.data > range_max] = np.nan

        if (~np.isfinite(rsi_ts.data)).any() and interpolate_inf:
            rsi_ts = rsi_ts.interpolate()

        return rsi_ts

    def to_xarray(self):
        '''Method to convert Timeseries
        to an xarray DataArray
        '''
        import xarray as xr

        return xr.DataArray(
            data=self.data,
            dims=['bands', 'time', 'x', 'y'],
            coords=dict(time=self.timestamps,
                        bands=self.bands))

    @classmethod
    def from_timeseries(cls, timeseries):
        return timeseries[0].merge(*timeseries[1:])

    def save(self, filename, compress=3):
        data = [self.data,
                self.timestamps,
                self.bands,
                self.attrs]
        joblib.dump(data, filename, compress=compress)

    @classmethod
    def load(cls, filename: str) -> 'Features':
        data = joblib.load(filename)
        return cls(data=data[0],
                   timestamps=data[1],
                   bands=data[2],
                   attrs=data[3])

    def __getitem__(self, band) -> 'Timeseries':
        return self.select_bands([band])

    def __repr__(self):
        return (f"<Timeseries: {self.data.shape} - bands: {self.bands} "
                f"- s: {self.timestamps[0]} - e: {self.timestamps[-1]} "
                f"- dtype: {self.data.dtype}>")

