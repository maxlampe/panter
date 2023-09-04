"""Base module for correcting data."""

import configparser

from panter.config import conf_path
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo

cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

# Import default ntof windows
BEAM_MEAS_TIME = {
    "sg": [
        float(cnf["dataPerkeo"]["BEAM_SIG_LOW"]),
        float(cnf["dataPerkeo"]["BEAM_SIG_UP"]),
    ],
    "bg": [
        float(cnf["dataPerkeo"]["BEAM_BG_LOW"]),
        float(cnf["dataPerkeo"]["BEAM_BG_UP"]),
    ],
}


class CorrBase:
    """Base class for doing corrections on PERKEO data.

    Takes a data loader output array and corrects all entries in it. CorrSimple and
    CorrPerkeo inherent from CorrBase.

    Parameters
    ----------
    dataloader: DLPerkeo
    bonlynew: True
        Only create corrected spectra instead of uncorrected spectra as well.
        Can be used for debugging and sanity checks.

    Attributes
    ----------
    corrections : {}
        Dictionary with bools to turn on/off different data corrections.
    histograms : []
        List to store created histograms to, if bstore=True in self.corr()
    hist_concat : HistPerkeo
        Concatenated histogram of all generated histograms, if bconcat is set to True
        in corr() method. Only works with mode=0 at the moment.
    addition_filters: []
        List of individual entries to be used as filters with data
        RootPerkeo.set_filt() in _filt_data().

    Examples
    --------
    See usage with CorrSimple (non-Energy data, e.g. ToF) and CorrPerkeo (Energy data).
    """

    def __init__(
        self,
        dataloader: DLPerkeo,
        bonlynew: bool = True,
    ):
        self._dataloader = dataloader
        self._bonlynew = bonlynew

        self._beam_mtime = BEAM_MEAS_TIME
        self.corrections = {}
        self.histograms = []
        self.hist_concat = None
        self.addition_filters = []

    def _set_corr(self):
        """Activate corrections from list."""
        pass

    def set_all_corr(self, bactive: bool):
        """Switch all corrections to active or inactive

        Parameters
        ----------
        bactive: bool
            Value to set all corrections to. (active = True)
        """

        for corr in self.corrections:
            self.corrections[corr] = bactive

        return 0

    def clear(self):
        """Clear relevant attributes enabling re-usability without new instantiation."""

        self.addition_filters = []
        self.histograms = []

        return 0

    def _filt_data(self, data: RootPerkeo, bbeam=False, key="", withauto: bool = True):
        """Filter data set.

        Parameters
        ----------
        data: RootPerkeo
            Data class to be filtered.
        bbeam: False
            Bool to apply nToF filter.
        key: ""
            Key to set signal or background nToF filter. Requires bbeam = True.
        withauto: True
            Filter data with RootPerkeo.auto(1)
        """

        if bbeam:
            data.set_filtdef()
            data.set_filt(
                "data",
                fkey="DeltaTriggerTime",
                active=True,
                ftype="num",
                low_lim=self._beam_mtime[key][0],
                up_lim=self._beam_mtime[key][1],
            )
            if len(self.addition_filters) != 0:
                for entry in self.addition_filters:
                    data.set_filt(**entry)
            if withauto:
                data.auto(1)
        else:
            if len(self.addition_filters) != 0:
                data.set_filtdef()
                for entry in self.addition_filters:
                    data.set_filt(**entry)
                if withauto:
                    data.auto(1)
            else:
                if withauto:
                    data.auto()

    def _calc_corr(self, data: RootPerkeo):
        """Calculate corrected amplitude for each event and file.

        Parameters
        ----------
        data: RootPerkeo
            Data class to be corrected.
        """
        pass

    def _corr_nobg(self, ev_file: list):
        """Correct measurement without background subtraction

        Parameters
        ----------
        ev_file: list
            File name of data file as a list with one entry.
        """
        pass

    def _corr_beam(self, ev_file: list):
        """Correct beam-like data (i.e. background in same file).

        Parameters
        ----------
        ev_file: list
            File name of data file as a list with one entry.
        """
        pass

    def _corr_src(self, ev_files: list):
        """Correct source-like data (i.e. background in different file).

        Parameters
        ----------
        ev_files: list
            Signal and background file names in a list.
        """
        pass

    def corr(self, bstore: bool = False, bwrite: bool = True, bconcat: bool = False):
        """Correcting data according to chosen settings.

        Parameters
        ----------
        bstore: False
            Bool whether to append created histograms in self.histograms
        bwrite: True
            Bool whether to write created histograms to a ROOT file.
        bconcat: False
            Bool to concatenate spectra.
        """
        pass
