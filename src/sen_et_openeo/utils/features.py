from typing import List
import sys
import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import scipy
from loguru import logger
from shapely.geometry import Point
from rasterio.crs import CRS
from skimage.transform import resize
import random
import string

from sen_et_openeo.utils.geotiff import get_rasterio_profile, write_geotiff


def random_string(n=8):
    x = ''.join(random.choice(string.ascii_uppercase +
                              string.ascii_lowercase +
                              string.digits) for _ in range(n))
    return x


def rasterize(gdf,
              bounds,
              epsg,
              resolution=10,
              value_column='Index',
              fill_value=np.nan,
              dtype=np.float32):
    """
    Rasterize a GeoDataFrame to a raster, by giving the bounds
    and epsg of the area to rasterize.
    `value_column` is the column to be used to fill the pixel values
    of each geometry. If none is provided, the index value of the
    geometry will be used.
    """
    out_shape = ((bounds[3] - bounds[1]) // resolution,
                 (bounds[2] - bounds[0]) // resolution)
    out_shape = list(map(int, out_shape))

    gs = gpd.GeoSeries(Polygon.from_bounds(*bounds),
                       crs=CRS.from_epsg(epsg)).to_crs(gdf.crs)
    geom = gs.geometry.values[0]

    geom_rows = gdf[gdf.intersects(geom)].copy()

    if geom_rows.shape[0] == 0:
        # empty gdf
        return np.ones(out_shape, dtype=dtype) * fill_value

    geom_buff = geom.buffer(0.01)
    geom_rows['inter_geometry'] = geom_rows.intersection(geom_buff)

    shapes = [(row.__getattribute__('inter_geometry'),
               row.__getattribute__(value_column)) for row
              in geom_rows.set_geometry('inter_geometry')
              .to_crs(epsg=epsg).itertuples()]

    transform = rasterio.transform.from_bounds(*bounds, *out_shape[::-1])
    try:
        raster = rasterio.features.rasterize(shapes,
                                             out_shape,
                                             fill=fill_value,
                                             transform=transform,
                                             dtype=dtype)
    except ValueError:
        # logger.warning(e)
        raster = np.ones(out_shape, dtype=dtype) * fill_value

    return raster


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def dem_attrs(dem_arr, attributes=['slope_riserun', 'aspect']):
    import richdem as rd
    rda = rd.rdarray(dem_arr, no_data=-9999)
    with HiddenPrints():
        attrs = [rd.TerrainAttribute(rda, attrib=attr) for attr in attributes]
    return attrs


def percentile_iqr(x, q=[10, 50, 90], iqr=[25, 75]):

    q_all = list(set(q + iqr))
    q_ids = [q_all.index(qv) for qv in q]
    iqr_ids = [q_all.index(qv) for qv in iqr]

    perc = np.percentile(x, q=q_all, axis=0)

    perc_arr = perc[q_ids]
    iqr_arr = perc[iqr_ids[1]] - perc[iqr_ids[0]]

    iqr_arr = np.expand_dims(iqr_arr, axis=0)

    arr = np.concatenate([perc_arr, iqr_arr], axis=0)

    return arr


def tsteps(x, n_steps=6):
    return scipy.signal.resample(x, n_steps, axis=0)


FEATURES_META = {
    "percentile_iqr": {
        "function": percentile_iqr,
        "parameters": {
            'q': [10, 50, 90],
            'iqr': [25, 75]
        },
        "names": ['p10', 'p50', 'p90', 'iqr']
    },
    "tsteps": {
        "function": tsteps,
        "parameters": {
            'n_steps': 6,
        },
        "names": ['ts0', 'ts1', 'ts2', 'ts3', 'ts4', 'ts5']
    }
}


class Features:

    def __init__(self,
                 data: np.ndarray,
                 names: List,
                 dtype: type = np.float32,
                 attrs: dict = dict()):

        if not isinstance(names, List):
            raise TypeError("`names` should be a list.")

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        if data.shape[0] != len(names):
            raise ValueError('axis 0 of `data` should be equal to '
                             'the length of `names`. Instead arr.shape = '
                             f'{data.shape} and len(names) = {len(names)}.')

        if data.dtype != dtype:
            self.data = data.astype(dtype)
        else:
            self.data = data

        self.names = names
        self.attrs = attrs
        self._df = None

    def _clone(self,
               data: np.ndarray = None,
               names: List = None,
               dtype: type = None,
               attrs: dict = None):

        data = self.data if data is None else data
        names = self.names if names is None else names
        dtype = self.data.dtype if dtype is None else dtype
        attrs = self.attrs if attrs is None else attrs

        return self.__class__(data, names, dtype, attrs)

    def get_feature_index(self, feature_name):
        return self.names.index(feature_name)

    def __getitem__(self, feature_name):
        idx = self.get_feature_index(feature_name)
        return self.data[idx]

    def __setitem__(self, feature_name, value):

        if np.isscalar(value):
            value = np.ones(self.shape[-2:]) * value

        if feature_name not in self.names:
            if value.shape[-2:] != self.data.shape[-2:]:
                raise ValueError("`value` should have the same shape "
                                 "of `self.data` on the last two axis.")
            new_feat = self._clone(value, [feature_name])
            self = self.merge(new_feat)

        else:
            idx = self.get_feature_index(feature_name)
            self.data[idx] = value

    def drop(self, *features_names):
        indices = [self.get_feature_index(f) for f in features_names]
        new_data = np.delete(self.data, indices, axis=0)
        new_names = [f for f in self.names if f not in features_names]
        return self._clone(new_data, new_names)

    @property
    def df(self):
        if self._df is None:
            self._df = self._arr_to_df()
        else:
            # If needed add attrs to df
            for key, value in self.attrs.items():
                if key not in self._df.columns:
                    self._df[key] = value
        return self._df

    @property
    def shape(self):
        return self.data.shape

    @df.setter
    def df(self, value):
        if isinstance(value, pd.DataFrame):
            self._df = value
        else:
            raise TypeError("Trying to set a non DataFrame type to `df`. "
                            f"type(value) = {type(value)}")

    def _arr_to_df(self):
        features_arr, features_names = self.data, self.names
        features_df = pd.DataFrame(
            features_arr.reshape(features_arr.shape[0],
                                 features_arr.shape[1] *
                                 features_arr.shape[2]).T,
            columns=features_names)

        # Convert attributes to columns
        for key, value in self.attrs.items():
            features_df[key] = value

        return features_df

    def downsample_categorical(self) -> 'Features':
        """
        Downsample categorical feature in the space dimensions.
        Based on majority vote within each 2x2 pixel window

        """
        from scipy.stats import mode

        old_data = self.data
        new_shape = (old_data.shape[0],
                     int(round(old_data.shape[1] / 2)),
                     int(round(old_data.shape[2] / 2)))

        new_data = np.empty(new_shape)
        for b in range(new_data.shape[0]):
            x = 0
            for i in range(new_data.shape[1]):
                y = 0
                for j in range(new_data.shape[2]):
                    new_data[b, i, j] = mode(old_data[b, x:x+2, y:y+2], axis=None).mode[0]
                    y += 2
                x += 2

        return self._clone(new_data, self.names)  # type: ignore

    def upsample(self, order=1, times=1, scaling=None) -> 'Features':
        """
        Upsamples Features in the space dimensions

        If order is 0: usese numba upsample implementation, doubles the
        resolution using 'nearest_neighbors' n 'times'. Ignores the 'scaling'
        parameter.

        If order is > 0: uses skimage.transform.rescale on every band
        (casting to float32 each time as rescale returns float64)
        If scaling is specified it will use that to determine the new shape,
        otherwise it will default to '2 ** times'.

        Order can be:
        0: Nearest-neighbor (usese numba custom, ignores scaling)
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

        """
        from sen_et_openeo.utils.resample import imresize, upsample

        old_data = self.data

        if order == 0:
            for t in range(times):
                new_data = upsample(old_data)
                old_data = new_data
        else:
            scaling = scaling or (2 ** times)

            new_shape = (old_data.shape[0],
                         int(round(old_data.shape[1] * scaling)),
                         int(round(old_data.shape[2] * scaling)))

            new_data = np.empty(new_shape)
            for i in range(new_data.shape[0]):
                new_data[i, ...] = imresize(old_data[i],
                                            shape=new_shape[1:],
                                            order=order)

        return self._clone(new_data, self.names)  # type: ignore

    def resize(self, scaling, order=1, anti_aliasing=True) -> 'Features':
        """
        Resize Features in the space dimensions

        Order can be:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

        """
        from sen_et_openeo.utils.resample import imresize

        old_data = self.data

        new_shape = (old_data.shape[0],
                     int(round(old_data.shape[1] * scaling)),
                     int(round(old_data.shape[2] * scaling)))

        new_data = np.empty(new_shape)
        for i in range(new_data.shape[0]):
            new_data[i, ...] = imresize(old_data[i],
                                        shape=new_shape[1:],
                                        order=order,
                                        anti_aliasing=anti_aliasing)

        return self._clone(new_data, self.names)

    def select(self, features_names, fill_value=None):

        features_names_valid = [f for f in features_names if f in self.names]
        features_names_invalid = [f for f in features_names
                                  if f not in self.names]

        if len(features_names_valid) == 0:

            if fill_value is not None:
                logger.warning("No valid features selected, returning "
                               f"constant features of value: {fill_value}")
                new_names = features_names
                new_shape = (len(features_names),
                             self.data.shape[1],
                             self.data.shape[2])
                new_data = np.ones(new_shape) * fill_value
                return self._clone(new_data, new_names)
            else:
                raise ValueError("No valid features selected")

        new_idxs = [self.names.index(f) for f in features_names_valid]
        try:
            new_data = self.data[new_idxs]
        except IndexError:
            # Temporary workaround for zarr features
            new_data = self.data.get_orthogonal_selection(new_idxs)

        if len(features_names_invalid):

            if fill_value is None:
                raise ValueError(f"Features {features_names_invalid} "
                                 "are not in the data.")
            else:
                logger.warning(f"Features {features_names_invalid} not in "
                               "data, replacing with "
                               f"constant features of value: {fill_value}")
                invalid_idxs = [features_names.index(f) for f in
                                features_names_invalid]
                for idx in invalid_idxs:
                    new_data = np.insert(new_data, idx, fill_value, axis=0)

        return self._clone(new_data, features_names)

    def merge(self, *others):
        """
        Returns a new instance merging data and names of current feature with
        multiple features.
        """
        new_names = self.names.copy()
        new_data = [self.data]

        for other in others:

            common_names = set(other.names) & set(self.names)
            if len(common_names) > 0:
                raise ValueError(f"Feature name: {common_names} "
                                 "already present. Cannot merge 'other'.")

            new_data.append(other.data)
            new_names.extend(other.names)

        new_data = np.concatenate(
            new_data,
            axis=0)

        return self._clone(new_data, new_names)

    def add(self,
            data: np.array,
            names: List):
        """
        Add an array with 1 or more features to the features stack
        """
        return self.merge(self.__class__(data, names))

    def add_constant(self,
                     constant_value,
                     constant_name):
        """
        Add a constant feature to the features array
        """
        if type(constant_value) is str:
            raise ValueError('`constant_value` should be numeric!')
        new_feat_arr = np.ones(self.shape[-2:]) * constant_value

        return self.merge(
            self._clone(new_feat_arr,
                        [constant_name]))

    def add_attribute(self,
                      attr_value,
                      attr_name):
        """
        Add an attribute to the features
        """
        new_attrs = self.attrs.copy()

        new_attrs[attr_name] = attr_value

        return self._clone(attrs=new_attrs)

    def cache_to_zarr(self,
                      filename=None,
                      chunks=(512, 512)):
        # Experimental Zarr caching
        import zarr

        if filename is None:
            filename = f'features_{random_string(8)}.zarr'

        logger.info(f'Caching features to zarr: {filename}')
        z = zarr.open(filename, mode='w',
                      shape=self.data.shape,
                      chunks=chunks, dtype=np.float32)

        z[:] = self.data
        self.data = z

    def add_geodataframe(self,
                         gdf,
                         feature_name,
                         bounds,
                         epsg,
                         resolution=10,
                         value_column='Index'
                         ):

        return self.merge(
            self.__class__.from_geodataframe(gdf,
                                             feature_name,
                                             bounds,
                                             epsg,
                                             resolution=resolution,
                                             value_column=value_column
                                             ))

    def add_l2a(self,
                l2a_collection,
                l2a_settings):

        return self.merge(
            self.__class__.from_l2a(l2a_collection,
                                    l2a_settings))

    def add_gamma0(self,
                   gamma0_collection,
                   gamma0_settings):
        return self.merge(
            self.__class__.from_gamma0(gamma0_collection,
                                       gamma0_settings))

    def add_agera5(self,
                   agera5_collection,
                   agera5_settings,
                   *args,
                   **kwargs):
        return self.merge(
            self.__class__.from_agera5(agera5_collection,
                                       agera5_settings,
                                       *args,
                                       **kwargs))

    def add_dem(self,
                dem_collection,
                dem_settings,
                resolution=10):

        return self.merge(
            self.__class__.from_dem(dem_collection,
                                    dem_settings,
                                    resolution=10))

    def add_latlon(self,
                   bounds,
                   epsg,
                   resolution=10):
        """
        Adds a lat, lon feature from the given bounds/epsg.

        See `from_latlon` for more details.

        """

        return self.merge(
            self.__class__.from_latlon(bounds,
                                       epsg,
                                       resolution=resolution))

    def add_pixelids(self):
        """
        Add a 'pixelids' with a different value for each pixel
        """
        new_shape = self.shape[-2:]
        new_feat_arr = np.arange(np.prod(new_shape)).reshape(new_shape)

        return self.merge(
            self._clone(new_feat_arr,
                        ['pixelids']))

    def onehot_encode(self, feature_name, prefix=None):
        """
        Returns an instance with the given the feature corresponding to
        `feature_name` onehot encoded
        """
        features = self

        if prefix is None:
            prefix = feature_name

        feat_arr = features[feature_name]
        new_features = features.drop(feature_name)

        values = np.unique(feat_arr)
        n = values.size

        encoded = np.zeros((n, *feat_arr.shape))
        encoded_names = []

        for i, v in enumerate(values):
            tmp = encoded[i]
            tmp[np.where(feat_arr == v)] = 1
            encoded[i] = tmp

            if np.isscalar(v):
                v = int(v)
            encoded_names.append(f"{prefix}_{v}")

        new_features = new_features.merge(Features(encoded, encoded_names))

        return new_features

    def to_geotiff(self, bounds, epsg, filename):
        profile = get_rasterio_profile(self.data, bounds, epsg)
        logger.debug(f"Saving {filename}...")
        write_geotiff(self.data, profile, filename, self.names)

    @ classmethod
    def from_features(cls, *features):
        return features[0].merge(*features[1:])

    @ classmethod
    def from_constant(cls, value, name, shape):
        """
        Return an instance with a constant value.
        """
        data = np.ones(shape[-2:]) * value
        return cls(data, [name])

    @ classmethod
    def from_geodataframe(cls,
                          gdf: gpd.GeoDataFrame,
                          feature_name,
                          bounds,
                          epsg,
                          resolution=10,
                          value_column='Index'):
        """
        Returns an instance from the rasterization of the `gdf`.
        `value_column` specifies the value for the geometries intersecting
        bounds and epsg given.
        The CRS of the `gdf` should be defined but doesn't need to be the same
        of epsg
        """
        geom_arr = rasterize(gdf,
                             bounds,
                             epsg,
                             resolution=resolution,
                             value_column=value_column)

        return cls(geom_arr, [feature_name])

    @ classmethod
    def from_l2a(cls,
                 l2a_collection,
                 l2a_settings,
                 rsi_meta=None,
                 features_meta=None,
                 ignore_def_features=False):

        return (l2a_collection
                .features_processor(l2a_settings,
                                    rsi_meta=rsi_meta,
                                    features_meta=features_meta,
                                    ignore_def_features=ignore_def_features)
                .compute_features())

    @ classmethod
    def from_l2a_seasons(cls,
                         l2a_collection,
                         l2a_settings,
                         rsi_meta=None,
                         features_meta=None,
                         ignore_def_features=False):

        return (l2a_collection
                .features_processor_seas(l2a_settings,
                                         rsi_meta=rsi_meta,
                                         features_meta=features_meta,
                                         ignore_def_features=ignore_def_features)  # noqa: E501
                .compute_features())

    @ classmethod
    def from_gamma0(cls,
                    gamma0_collection,
                    gamma0_settings,
                    rsi_meta=None,
                    features_meta=None,
                    ignore_def_features=False):

        return (gamma0_collection
                .features_processor(gamma0_settings,
                                    rsi_meta=rsi_meta,
                                    features_meta=features_meta,
                                    ignore_def_features=ignore_def_features)
                .compute_features().upsample())

    @ classmethod
    def from_sigma0(cls,
                    sigma0_collection,
                    sigma0_settings,
                    rsi_meta=None,
                    features_meta=None,
                    ignore_def_features=False):

        return (sigma0_collection
                .features_processor(sigma0_settings,
                                    rsi_meta=rsi_meta,
                                    features_meta=features_meta,
                                    ignore_def_features=ignore_def_features)
                .compute_features().upsample())

    @ classmethod
    def from_agera5(cls,
                    agera5_collection,
                    agera5_settings,
                    demcol=None,
                    bounds=None,
                    epsg=None,
                    features_meta={},
                    rsi_meta={},
                    ignore_def_features=True):

        return (agera5_collection
                .features_processor(agera5_settings,
                                    demcol=demcol,
                                    bounds=bounds,
                                    epsg=epsg,
                                    features_meta=features_meta,
                                    rsi_meta=rsi_meta,
                                    ignore_def_features=ignore_def_features)
                .compute_features().upsample())

    @ classmethod
    def from_dem(cls,
                 dem_collection,
                 dem_settings=None,
                 resolution=10):

        default_names = ['DEM-alt-20m', 'DEM-slo-20m',
                         'DEM-nor-20m', 'DEM-eas-20m']

        dem_settings = dem_settings or {}
        features_names = dem_settings.get('features_names',
                                          default_names)

        altitude = dem_collection.load().astype(np.float32)

        altitude[altitude < -10000] = np.nan

        slope, aspect = dem_attrs(altitude)
        aspect = np.deg2rad(aspect)
        northness = np.cos(aspect)
        eastness = np.sin(aspect)

        default_arrs = [altitude, slope, northness, eastness]

        dem_features_dict = {fn: fa for fn, fa in zip(default_names,
                                                      default_arrs)}

        dem_features_arr = np.array(
            [dem_features_dict[fn] for fn in features_names])

        feats = cls(dem_features_arr, features_names)

        if resolution == 10:
            feats = feats.upsample()
        elif resolution == 20:
            pass
        else:
            raise ValueError("`resolution` should be 10 or 20.")

        return feats

    @ classmethod
    def from_worldcover(cls,
                        worldcover_collection,
                        resolution=10):

        features_names = ['WORLDCOVER-LABEL-10m']

        label = worldcover_collection.load().astype(np.uint8)

        feats = cls(label, features_names, dtype=np.uint8)

        if resolution == 20:
            feats = feats.downsample_categorical()
        elif resolution == 10:
            pass
        else:
            raise ValueError("`resolution` should be 10 or 20.")

        return feats

    @ classmethod
    def from_latlon(cls,
                    bounds,
                    epsg,
                    resolution=10,
                    steps=5):
        """
        Returns a lat, lon feature from the given bounds/epsg.

        This provide a coarse (but relatively fast) approximation to generate
        lat lon layers for each pixel.

        'steps' specifies how many points per axis should be use to perform
        the mesh approximation of the canvas
        """

        xmin, ymin, xmax, ymax = bounds
        out_shape = (int(np.floor((ymax - ymin) / resolution)),
                     int(np.floor((xmax - xmin) / resolution)))

        xx = np.linspace(xmin + resolution/2, xmax + resolution/2, steps)
        yy = np.linspace(ymax + resolution/2, ymin + resolution/2, steps)

        xx = np.broadcast_to(xx, [steps, steps]).reshape(-1)
        yy = np.broadcast_to(yy, [steps, steps]).T.reshape(-1)

        points = [Point(x0, y0) for x0, y0 in zip(xx, yy)]

        gs = gpd.GeoSeries(points, crs=CRS.from_epsg(epsg))
        gs = gs.to_crs(epsg=4326)

        lon_mesh = gs.apply(lambda p: p.x).values.reshape((steps, steps))
        lat_mesh = gs.apply(lambda p: p.y).values.reshape((steps, steps))

        lon = resize(lon_mesh, out_shape, order=1, mode='edge')
        lat = resize(lat_mesh, out_shape, order=1, mode='edge')

        features_arr = np.array([lat, lon])
        features_names = ['lat', 'lon']

        return cls(features_arr, features_names)

    def __repr__(self):
        attrs_repr = ', '.join([f"{k}: {v}" for k, v in self.attrs.items()])
        if len(attrs_repr):
            attrs_repr = " - " + attrs_repr

        return (f"<Features: {self.data.shape}{attrs_repr}>")

    def save(self, filename, compress=3):
        data = [self.data,
                self.names,
                self.attrs]
        joblib.dump(data, filename, compress=compress)

    @ classmethod
    def load(cls, filename: str) -> 'Features':
        data = joblib.load(filename)
        return cls(data[0],
                   names=data[1],
                   attrs=data[2])
