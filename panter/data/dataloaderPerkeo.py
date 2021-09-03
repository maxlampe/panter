"""Data loader for accessing and filtering data to pass into advance analysis code."""

import numpy as np
import pandas as pd

from panter.data.dataFiles import sort_files
from panter.data.dataMeasPerkeo import MeasPerkeo


class DLPerkeo:
    """Data loader class for preparing list of files to evaluate.

    Stores events either by adding them manually to the loader or by automatically
    getting all from a directory. Beam time measurements are automatically sorted
    and calibration/drift measurements paired with their background measurement.
    Works on every directory with Perkeo III 19/20 automatic measurement files.

    Parameters
    ----------
    dir_name : str
        Name of directory to use the data loader in.

    Attributes
    ----------
    _dir_name : str
    _measurements : list of MeasPerkeo

    Examples
    --------
    General example of how to use DLPerkeo. The two cases demonstrate either the
    automatic fill function or manually adding events by a list. Event format is given
    by MeasPerkeo class.

    >>> dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    >>> dataloader = DLPerkeo(dir)
    >>> if True:
            dataloader.auto()
    >>> else:
            events = [MeasPerkeo(0, 5, list(['dir/data11-22_beam.root']), 11, 22),
                      MeasPerkeo(0, 7, list(['dir/data.root']))]
            dataloader.fill(events)
    >>> dataloader.print()
    >>> print(dataloader[0]())
    >>> all_meas = dataloader.ret_meas()
    >>> filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 2])
    """

    def __init__(self, dir_name: str):
        self._dir_name = dir_name
        self._measurements = []
        self.df = None

    def __len__(self):
        return len(self._measurements)

    def __getitem__(self, item):
        return self._measurements[item]

    def length(self):
        """Return length per measurement type. 0 if no measurements in data loader."""

        length = 0
        if len(self._measurements) != 0:
            found_types = [self._measurements[0].tp]
            counter = [0]
            for meas in self._measurements:
                curr_type = meas.tp
                if curr_type in found_types:
                    counter[found_types.index(curr_type)] += 1
                else:
                    found_types.append(curr_type)
                    counter.append(1)

            length = np.array([found_types, counter]).T

        return length

    def add_meas(self, event: MeasPerkeo):
        """"Add measurement manually to the list with vague validity check."""

        self._measurements.append(event)

        return 0

    def rem_meas(self, meas_no: list):
        """Remove given meas (by index) in meas_no list."""

        for i, no in enumerate(meas_no):
            del self._measurements[no - i * 1]

        return 0

    def fill(self, liste):
        """Add measurements from list."""

        for elem in liste:
            self.add_meas(MeasPerkeo(*elem))

        return 0

    def print(self, rng: list = None, bfilename=True):
        """Print a sample of measurements by list of positions."""

        if rng is not None:
            print(f"Data loader Entries with index {rng}")
            for no in rng:
                if bfilename:
                    print(no, self._measurements[no]())
                else:
                    print(no, np.asarray(self._measurements[no]())[0:2])
        else:
            print(f"All data loader Entries")
            for no, entry in enumerate(self._measurements):
                if bfilename:
                    print(no, entry())
                else:
                    print(no, np.asarray(entry())[0:2])

        return 0

    def ret_meas(self, rng: list = None) -> np.array:
        """Return measurements as array for given list of positions."""

        if rng is not None:
            return np.asarray(self._measurements)[rng]
        else:
            return np.asarray(self._measurements)

    def auto(self, bclear: bool = True):
        """Use dataFiles.py sort_files and import all."""

        if bclear:
            self._measurements = []
        self.fill(sort_files(self._dir_name))

        return 0

    def ret_df(self) -> pd.DataFrame:
        """Return measurements as DataFrame."""

        if self.df is None:
            meas0 = self._measurements[0].__dict__
            df_all = pd.DataFrame(columns=meas0.keys(), index=range(len(self)))
            for num, meas in enumerate(self._measurements):
                df_all.loc[num] = pd.Series(meas.__dict__)
            self.df = df_all

        return self.df

    def ret_filt_meas(self, key: list, val: list):
        """Return measurements filtered by MeasPerkeo attributes as array."""

        self.ret_df()
        curr_df = self.df
        for n, key in enumerate(key):
            filt = curr_df[key] == val[n]
            curr_df = curr_df[filt]

        filt_meas = []
        for index, row in curr_df.iterrows():
            dick = row.to_dict()
            del dick["date_list"]
            filt_meas.append(MeasPerkeo(**dick))

        return np.asarray(filt_meas)
