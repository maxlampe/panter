"""Master class for studying different phenomena over time."""

import numpy as np
import panter.core.dataPerkeo as dP
from panter.config import conf_path


class MapPerkeo:
    """Master class for studying different phenomena over time.

    Parameters
    ----------
    fmeas: np.array(MeasPerkeo)
        Array of data loader output (DLPerkeo).
    level: 0
        Number of levels of maps. E.g., for drift calculations,
        top level = final PMT factor map
        second level = Sn drift measurement peak fits
    bimport: list of bool
        Import settings for existing maps.

    Attributes
    ----------
    maps: list of pd.Dataframe
        List of pandas DataFrame with map values with a time stamp.
        Need to be imported or calculated.
    """

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        level: int = 1,
        bimport: list = None,
    ):
        self._fmeas = fmeas
        self._level = level
        if bimport is None:
            self._bimport = [False] * self._level
            self._bimport[0] = True
        else:
            self._bimport = bimport
        self.maps = [None] * self._level
        self.cache = None

    def __call__(self, *args, **kwargs):
        for level, bimp in enumerate(self._bimport):
            res = self._get_level(level=level, bimp=bimp)
            if res:
                break

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""
        return 0

    def _write_map2file(self, map_ind: int = 0, fname: str = "map.p"):
        """Write given map and cache into file in conf directory"""

        outfile = dP.FilePerkeo(fname)
        return outfile.dump([self.maps[map_ind], self.cache], conf_path)
