from typing import List
from abc import (ABC,
                 abstractmethod,
                 abstractproperty)

from datetime import datetime
from pathlib import Path
import re
import os
import concurrent.futures

import pandas as pd

from shapely.geometry import Polygon
import geopandas as gpd

from rasterio.crs import CRS

from sen_et_openeo.utils.geoloader import ParallelLoader


def is_s2esa_product(filename):
    basename = os.path.basename(filename)
    pattern = (r'^S2[AB]_MSIL(1C|2A)_\d{8}T\d{6}_N\d{4}_R\d{3}_'
               + r'T\d{2}[A-Z]{3}_\d{8}T\d{6}.(SAFE|zip)$')
    if re.match(pattern, basename):
        return True
    else:
        return False


def is_s2maja_product(filename):
    basename = os.path.basename(filename)
    maja_pattern = (r'^SENTINEL2[AB]_\d{8}-\d{6}-\d{3}_L2A_'
                    r'T\d{2}[A-Z]{3}')
    if re.match(maja_pattern, basename):
        return True
    else:
        return False


def is_s2aws_product(filename):
    basename = os.path.basename(filename)
    pattern = (r'^S2[AB]_MSIL(1C|2A)_\d{8}T\d{6}_N\d{4}_R\d{3}_'
               + r'T\d{2}[A-Z]{3}_\d{8}T\d{6}$')
    if re.match(pattern, basename):
        return True
    else:
        return False


def is_s2icor_product(filename):
    basename = os.path.basename(filename)
    icor_pattern = (r'^S2[AB]_MSIL1C_\d{8}T\d{6}_N\d{4}_R\d{3}_'
                    + r'T\d{2}[A-Z]{3}_\d{8}T\d{6}_iCOR$')
    if re.match(icor_pattern, basename):
        return True
    else:
        return False
    # raise NotImplementedError
    
def is_sentinel2_product(filename):
    """Check if file has L1C/L2A name and returns True or False"""
    if not isinstance(filename, str):
        raise TypeError("Input should be of string type")

    if (is_s2esa_product(filename)
            or is_s2maja_product(filename)
            or is_s2icor_product(filename)
            or is_s2aws_product(filename)):

        return True
    else:
        return False


def glob_sentinel2_products(path):
    """
    Generator of sentinel2 products filenames from directory.
    Scans recursively for .zip and .SAFE products
    """
    root_dir, folders, files = next(os.walk(path))

    for f in files:
        if is_sentinel2_product(f):
            yield os.path.join(root_dir, f)

    for d in folders:
        new_path = os.path.join(root_dir, d)
        if is_sentinel2_product(d):
            yield new_path
        else:
            yield from glob_sentinel2_products(new_path)

def s2esa_entry(filename):
    """
    Returns dictionary of sentinel 2 product with specifications from filename
    """
    basename = os.path.basename(filename)
    product_id, ext = basename.split('.')
    mission_id, level, date, baseline, orbit, tile, _ = product_id.split('_')

    entry = dict(level=level[3:],
                 product_id=product_id,
                 mission_id=mission_id,
                 date=datetime.strptime(date, '%Y%m%dT%H%M%S'),
                 baseline=baseline,
                 orbit=orbit,
                 tile=tile[1:],
                 zipped=(True if ext == 'zip' else False),
                 path=filename)

    return entry


def s2maja_entry(filename):
    """
    Returns dictionary of MAJA sentinel 2 product with specifications
    from filename
    """
    basename = os.path.basename(filename)
    product_id = basename
    mission_id_ext, date_ext, level, tile, _, _ = product_id.split('_')
    mission_id = 'S' + mission_id_ext[-2:]

    entry = dict(level=level,
                 product_id=product_id,
                 mission_id=mission_id,
                 date=datetime.strptime(date_ext[:-4], '%Y%m%d-%H%M%S'),
                 baseline=None,
                 orbit=None,
                 tile=tile[1:],
                 zipped=False,
                 path=filename)

    return entry


def s2aws_entry(filename):
    """
    Returns dictionary of sentinel 2 product with specifications from filename
    """
    basename = os.path.basename(filename)
    product_id = basename
    mission_id, level, date, baseline, orbit, tile, _ = product_id.split('_')

    entry = dict(level=level[3:],
                 product_id=product_id,
                 mission_id=mission_id,
                 date=datetime.strptime(date, '%Y%m%dT%H%M%S'),
                 baseline=baseline,
                 orbit=orbit,
                 tile=tile[1:],
                 path=filename)

    return entry


def s2idepix_entry(filename):
    """
    Returns dictionary of sentinel 2 product with specifications from filename
    """
    basename = os.path.basename(filename)
    product_id = basename
    mission_id, level, date, baseline, orbit, tile, _ = product_id.split('_')

    entry = dict(level=level[3:],
                 product_id=product_id[:-4],
                 mission_id=mission_id,
                 date=datetime.strptime(date, '%Y%m%dT%H%M%S'),
                 baseline=baseline,
                 orbit=orbit,
                 tile=tile[1:],
                 path=filename)

    return entry


def s2icor_entry(filename):
    """
    Returns dictionary of sentinel 2 product with specifications from filename
    """
    basename = os.path.basename(filename)
    product_id = basename
    mission_id, level, date, baseline, orbit, tile, _, _ = product_id.split(
        '_')

    entry = dict(level=level[3:],
                 product_id=product_id,
                 mission_id=mission_id,
                 date=datetime.strptime(date, '%Y%m%dT%H%M%S'),
                 baseline=baseline,
                 orbit=orbit,
                 tile=tile[1:],
                 path=filename)

    return entry

def sentinel2_entry(filename):
    if is_s2esa_product(filename):
        return s2esa_entry(filename)
    elif is_s2maja_product(filename):
        return s2maja_entry(filename)
    elif is_s2aws_product(filename):
        return s2aws_entry(filename)
    elif is_s2icor_product(filename):
        return s2icor_entry(filename)
    else:
        raise ValueError('filename: {} is not a valid sentinel2 product'
                         .format(filename))

def _build_s2products_df(folder, threads=20):
    """
    Globs sentinel2 products in folder (SAFE and ZIP) and
    returns a Pandas dataframe with the filenames and basic metadata
    info
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        products = list(
            ex.map(lambda x: x, glob_sentinel2_products(folder)))
    # products = list(glob_sentinel2_products(folder))

    entries = [sentinel2_entry(f) for f in products]
    if len(entries):
        df = pd.DataFrame(entries)
    else:
        df = pd.DataFrame([], columns=['level',
                                       'product_id',
                                       'mission_id',
                                       'date',
                                       'baseline',
                                       'orbit',
                                       'tile',
                                       'zipped',
                                       'path'])

    return df


def build_s2products_df(*folders):
    """
    Globs sentinel2 products in given folders (SAFE and ZIP) and
    returns a Pandas dataframe with the filenames and basic metadata
    info
    """
    dfs_list = [_build_s2products_df(folder) for folder in folders]
    return pd.concat(dfs_list, axis=0)

def build_products_df(sensor, *folders):
    if sensor == 'S2':
        return build_s2products_df(*folders)
    else:
        raise NotImplementedError(f"Unrecognized sensor: {sensor}")


class BaseCollection(ABC):
    """
    Abstract class describing the base methods of a collection class
    """

    @abstractproperty
    def supported_bands(self):
        pass

    @abstractproperty
    def supported_resolutions(self):
        pass

    @abstractproperty
    def bands(self):
        pass

    @abstractproperty
    def products(self):
        pass

    @abstractproperty
    def timestamps(self):
        pass

    @abstractproperty
    def loader(self):
        pass

    @abstractmethod
    def get_band_filenames(self, band, resolution):
        pass


class DiskCollection(BaseCollection):
    """
    Abstract class base for all collections of files stored on a local
    filesystem. The products must be tiled using the Sentinel-2 grid
    """

    def __init__(self, df, s2grid=None):

        import sen_et_openeo
        
        if df.empty:
            df = self._empty_df()

        self.df = df.sort_values('date')
        self.tiles = sorted(self.df.tile.unique().tolist())
        self.start_date = datetime(2000, 1, 1)
        self.end_date = datetime(2100, 1, 1)

        self._s2grid = (s2grid if s2grid is not None
                        else sen_et_openeo.layers.load('s2grid'))

        self._bounds = None
        self._filenames = None
        self._bands = None
        self._epsg = None
        self._loader = None
        self._sensor = NotImplementedError

    @classmethod
    def from_path(cls, path, s2grid=None):
        path = Path(path)
        if path.is_file():
            return cls.from_file(path, s2grid=s2grid)
        elif path.is_dir():
            return cls.from_folders(path, s2grid=s2grid)
        else:
            raise ValueError(f'{path} is neither a folder nor a file.')

    @classmethod
    def from_file(cls, filename, s2grid=None):
        df = pd.read_csv(filename)
        df.date = pd.to_datetime(df.date)
        return cls(df, s2grid=s2grid)

    @classmethod
    def from_folders(cls, *folders, s2grid=None):
        df = build_products_df(cls.sensor, *folders)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    def save(self, filename):
        self.df.to_csv(filename, index=False)

    @property
    def epsg(self):
        return self._epsg

    @property
    def bounds(self):
        return self._bounds

    @property
    def bands(self):
        if self._bands is None:
            self.bands = self.supported_bands
        return self._bands

    @bands.setter
    def bands(self, bands):
        for b in bands:
            if b not in self.supported_bands:
                raise ValueError("Band {} not supported.".format(b))
        self._bands = bands

    @property
    def products(self):
        return self.df.path.values.tolist()

    @property
    def timevector(self):
        return self.df.date.values

    @property
    def timestamps(self):
        return self.df.date.apply(lambda x: str(x)).values.tolist()

    @property
    def loader(self):
        if self._loader is None:
            self._loader = ParallelLoader()
        return self._loader

    @loader.setter
    def loader(self, value):
        self._loader = value

    def filter_dates(self, start_date, end_date):
        df = self.df[(self.df.date >= start_date)
                     & (self.df.date < end_date)]
        return self._clone(df=df, start_date=start_date, end_date=end_date)

    def filter_bounds(self, bounds, epsg):
        self._epsg = int(epsg)
        self._bounds = bounds
        products = self._get_products(self._bounds, self._epsg)
        return self._clone(df=products)

    def filter_bands(self, *bands):
        return self._clone(bands=bands)

    def filter_tiles(self, *tiles):
        df = self.df[self.df.tile.isin(tiles)]
        return self._clone(df=df)

    def _get_products(self,
                      bounds: List,
                      epsg: int) -> pd.DataFrame:
        """
        Returns subset of products df that intersect with the given
        bounds and EPSG
        """
        gs = gpd.GeoSeries(Polygon.from_bounds(*bounds),
                           crs=CRS.from_epsg(epsg)).to_crs(epsg=4326)
        bbox = gs.iloc[0]

        tiles = self._s2grid[self._s2grid.intersects(bbox)].tile

        if tiles.size == 0:
            raise ValueError("No products available for the specified bounds "
                             " and EPSG.")

        products = self.df[self.df.tile.isin(tiles)]

        products = products.sort_values('date',
                                        ascending=True)

        # drop data duplicated in overlapping zones
        # products = products_sorted.drop_duplicates(subset=['date'])
        return products

    def _clone(self,
               df=None,
               bands=None,
               start_date=None,
               end_date=None):
        """
        Returns an instance of self with updated parameters
        """
        new_collection = (self.__class__(self.df, self._s2grid) if df is None
                          else self.__class__(df, self._s2grid))
        new_collection.bands = self.bands if bands is None else bands
        new_collection._bounds = self._bounds
        new_collection._epsg = self._epsg
        new_collection._loader = self._loader
        new_collection.start_date = (self.start_date if start_date is None
                                     else start_date)
        new_collection.end_date = (self.end_date if end_date is None
                                   else end_date)

        return new_collection

    def load(self,
             bands=None,
             resolution=None,
             loader=None,
             resample=False):

        resolution = (resolution if resolution is not None
                      else self.supported_resolutions[0])

        if resolution not in self.supported_resolutions:
            raise NotImplementedError("Given resolution is not supported.")

        bands = bands if bands is not None else self.bands

        data_loader = loader if loader is not None else self.loader

        data = data_loader.load(self, bands, resolution,
                                resample=resample)

        return data

    def load_timeseries(self,
                        *bands,
                        **kwargs):
        from sen_et_openeo.utils.timeseries import load_timeseries
        return load_timeseries(self, *bands, **kwargs)

    @staticmethod
    def _empty_df():
        columns = ['level', 'product_id', 'date', 'tile', 'path']
        df = pd.DataFrame([], columns=columns)
        return df
