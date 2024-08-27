from abc import ABC, abstractproperty

from satio.rsindices import RSI_META
from satio.utils import TaskTimer


class _TimeSeriesTimer():

    def __init__(self, *resolutions):

        self.load = {}
        self.rsi = {}
        self.composite = {}
        self.interpolate = {}

        for r in resolutions:
            self.load[r] = TaskTimer(f'{r}m loading')
            self.rsi[r] = TaskTimer(f'{r}m rsi calculation')
            self.composite[r] = TaskTimer(f'{r}m compositing')
            self.interpolate[r] = TaskTimer(f'{r}m interpolation')


class TimeSeriesProcessor(ABC):

    def __init__(self,
                 collection,
                 settings,
                 rsi_meta=None):

        self.collection = collection
        self.settings = settings
        self.sensor = collection.sensor
        self._bands = None
        self._rsis = None
        self._supported_bands = None
        self._supported_rsis = None

        # merge default RSI_META with provided ones
        self._rsi_meta = {**RSI_META.get(self.sensor, {}),
                          **(rsi_meta or {})}

    @ abstractproperty
    def supported_bands(self):
        ...

    @ abstractproperty
    def supported_rsis(self):
        ...

    @ property
    def bands(self):
        if self._bands is None:
            self._bands = {res: [b for b in self.settings['bands']
                                 if b in sup_bands]
                           for res, sup_bands in self.supported_bands.items()}
        return self._bands

    @ property
    def rsis(self):
        if self._rsis is None:
            self._rsis = {res: [b for b in self.settings.get('rsis', [])
                                if b in sup_bands]
                          for res, sup_bands in self.supported_rsis.items()}
        return self._rsis
