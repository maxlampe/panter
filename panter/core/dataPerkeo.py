"""Module for handling Perkeo root files and creating histograms."""

from __future__ import annotations

import sys
import os
import time
import configparser
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import uproot

from panter.config import conf_path


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")


def ret_hist(
    data: np.array(float),
    bincount: int = 1024,
    range_lowlim: int = 0,
    range_uplim: int = 52000,
):
    """Create a histogram as pd.DataFrame from an input array."""

    hist, binedge = np.histogram(data, bincount, (range_lowlim, range_uplim))
    bincent = []
    for j in range(binedge.size - 1):
        bincent.append(0.5 * (binedge[j] + binedge[j + 1]))

    return pd.DataFrame({"x": bincent, "y": hist, "err": np.sqrt(np.abs(hist))})


def filt_zeros(histdf: pd.DataFrame) -> pd.DataFrame:
    """"""

    filt = histdf["y"] == 0.0
    histdf = histdf[filt]

    return histdf


class HistPerkeo:
    """Histogram object for use with PERKEO data.

    Takes data and histogram parameters to create a histogram with
    basic helper functions.

    Parameters
    ---------
    data : array of float
    bin_count, low_lim, up_lim: int
        Histogram parameters: Bin count, upper and lower limit

    Attributes
    ----------
    data, bin_count, low_lim, up_lim
        see above section
    hist : pd.DataFrame
        Returned histogram from function ret_hist()
    """

    def __init__(
        self,
        data: np.array(float),
        bin_count: int = 1024,
        low_lim: int = 0,
        up_lim: int = 52000,
    ):
        self.data = data
        self.bin_count = bin_count
        self.up_lim = up_lim
        self.low_lim = low_lim
        self.hist = ret_hist(self.data, self.bin_count, self.low_lim, self.up_lim)

    def plt(
        self,
        rng: list = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plot histogram."""

        figure(figsize=(8, 6))
        plt.errorbar(self.hist["x"], self.hist["y"], self.hist["err"], fmt=".")
        if rng is not None:
            plt.axis([rng[0], rng[1], rng[2], rng[3]])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.annotate(
            f"Mean = {np.array(self.data).mean():0.2f}\n"
            f"StDv = {np.array(self.data).std():0.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="1"),
        )
        plt.show()

        return 0

    def addhist(self, hist_p: HistPerkeo, fac: float = 1.0):
        """Add another histogram to existing one with multiplicand."""

        hist_par = [self.bin_count, self.low_lim, self.up_lim]
        hist_par_2add = [hist_p.bin_count, hist_p.low_lim, hist_p.up_lim]
        assert hist_par == hist_par_2add, "ERROR: Binning does not match."

        newhist = pd.DataFrame(
            {
                "x": self.hist["x"],
                "y": (self.hist["y"] + fac * hist_p.hist["y"]),
                "err": np.sqrt(self.hist["err"] ** 2 + (fac * hist_p.hist["err"]) ** 2),
            }
        )
        # Changes input ret_hist like in Root
        self.hist = newhist

        return 0

    def ret_asnumpyhist(self):
        """Return histogram in np.histogram format from current histogram.

        Note: Cannot use data, as this doesnt include added histograms etc."""

        deltx = 0.5 * (self.hist["x"].values[1] - self.hist["x"].values[0])
        binedge = self.hist["x"].values - deltx
        binedge = np.append(binedge, self.hist["x"].values[-1] + deltx)

        return self.hist["y"].values, binedge


class FilePerkeo:
    """Obj for writing/reading data from/into a general binary file."""

    def __init__(self, filename: str):
        self.filename = filename

    def imp(self):
        """Open the file and return content."""
        with open(self.filename, "rb") as file:
            imp_obj = pickle.load(file)

        return imp_obj

    # FIXME: Append doesn't append? Probsbecause import only expects one object
    def dump(self, obj, bapp: bool = False, btext: bool = False):
        """Dump an python object into the file.

        Parameters
        ----------
        obj
        bapp : bool
            Append to file instead of writing over it. Doesn't work.
            (default False)
        btext : bool
            Bool whether to dump obj as binary or text into file.
            (default False)
        """

        opt = "a" if bapp else "w"
        if not btext:
            with open(self.filename, opt + "b") as file:
                pickle.dump(obj, file)
        else:
            with open(self.filename, opt) as file:
                file.write(obj)

        return 0


class DirPerkeo:
    """Obj for extracting file name lists out of directory."""

    def __init__(self, dirname: str, filetype: str = "root"):
        self.dirname = dirname
        self.filetype = filetype

    def get_all(self, bsorted=True) -> list:
        """Returns list of all files of given type in directory"""

        if bsorted:
            liste = sorted(glob.glob(self.dirname + "*." + self.filetype))
        else:
            liste = glob.glob(self.dirname + "*." + self.filetype)

        return liste

    def get_subset(self, liste: list(str)) -> list(str):
        """Returns list of required files (full name) out of list in dir"""

        retlist = []
        for i in liste:
            retlist.append(glob.glob(self.dirname + i)[0])

        return retlist


class FiltPerkeo:
    """Obj for creating/setting filters for RootPerkeo Trees.

    Attributes
    ----------
    active : bool
        Status bool whether filter is active
    type : str
        Filter type for safety handling: 'bool', 'num' or special
        type 'cyc'
    lowerlimit, upperlimit: float
        Filter range for 'num' type filter
    rightval
        Value for 'bool' type filter to check quality for
    """

    def __init__(self, active: bool, ftype: str):
        self.active = active
        self.ftype = ftype
        self.lowerlimit = None
        self.upperlimit = None
        self.rightval = None


class RootPerkeo:
    """Class for top layer PERKEO root file management.

    Takes filename and can generate data histograms according to set
    filters. Currently only for all PMTs individually. Has several helper functions.

    Parameters
    ----------
    filename : str

    Attributes
    ----------
    filename : str
    filedate : float
        Time (in seconds) since epoch of last modification of file.
    file : uproot.open()
        Raw root file in python via uproot package.
    mode : {1, 2, 3}
        Measurement program mode. (1, 2, 3) -> (Delta, Both, All)
    _tadc : {0, 1, 32}
        Depending on mode, is max _tadc index value of integrator
    no_pmts : int
        Is set automatically. Should be 16 (e.g. LogicBox would be 4).
    _ev_no, _cy_no : int
        Number of events and cycles respectively.
    _ev_valid, _cy_valid : array of bool
        Length of array is either _ev_no or _cy_no. Is set depending on
        filter settings. 1/True is valid and 0/False would be invalid.
    _ev_valid_no, cy_valid_no : int
        Number of valid events and valid cycles respectively.
    cyclefilter, datafilter: dict of FiltPerkeo
    pmt_data : list of arrays
        List of arrays with all PMT data after generation with filters
    dptt : array
        Array to store DeltaPrevTriggerTime values for files without it.
    stats : dict of lists
        Dictionary for PMT statistics: mean, stddev, max xval, 'active'
        for each PMT (array in pmt_data).
    _pmt_thres : int
        Virtual threshhold for mean of a PMT data array to be
        considered 'active'. Has no effect outside this label. Is for
        automated fits and set by ini file.
    hists : list of HistPerkeo()
        List of histograms for each PMT. Need to be initialized and generated.
    _hist_par : list of int
        Histogram default parameters imported by ini file (evalRaw.ini)
        ['ADC_hist_counts', 'ADC_hist_min', 'ADC_hist_max']
    hist_sums : list of HistPerkeo()
        List of histograms for DetSum 0 and 1. Need to be initialized and generated.
    _histsum_par  : list of int
        Histogram default parameters imported by ini file (evalRaw.ini)
        ['SUM_hist_counts', 'SUM_hist_min', 'SUM_hist_max']

    Examples
    --------
    General example of how to use RootPerkeo. The two cases differ by
    using default or custom filters (here DeltaPrevTriggerTime).
    Histograms are created only for 'active' PMTs in this example.

    >>> data = dP.RootPerkeo("file.root")
    >>> data.info()
    >>> if True:
            data.auto()
        else:
            data.set_filtdef()
            data.datafilter['DeltaTriggerTime'].active = True
            data.datafilter['DeltaTriggerTime'].upperlimit = ()
            data.datafilter['DeltaTriggerTime'].lowerlimit = ()
            data.auto(1)
    >>> data.gen_hist(data.ret_actpmt())
    >>> dtt = 0.01 * data.ret_array_by_key("DeltaTriggerTime") # [mus]
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.filedate = os.path.getmtime(filename)
        self.file = uproot.open(self.filename)
        self.mode = None
        self._tadc = None
        self.no_pmts = None
        self._ev_no = len(self.file["dataTree"].array("EventNumber"))
        self._cy_no = len(self.file["cycleTree"].array("Cycle"))
        self._bDPTT = b"DeltaPrevTriggerTime" in self.file["dataTree"].keys()
        self._ev_valid = None
        self._cy_valid = None
        self._ev_valid_no = None
        self.cy_valid_no = None

        self.cyclefilter = {
            "Cycle": FiltPerkeo(False, "num"),
            "Valid": FiltPerkeo(True, "bool"),
            "ChopperSpeed": FiltPerkeo(False, "num"),
        }
        self.datafilter = {
            "Cycle": FiltPerkeo(True, "cyc"),
            "PMT": FiltPerkeo(False, "num"),
            "DetSum": FiltPerkeo(False, "num"),
            "Detector": FiltPerkeo(False, "bool"),
            "CoinTime": FiltPerkeo(False, "num"),
            "TDC": FiltPerkeo(False, "num"),
            "DeltaPrevTriggerTime": FiltPerkeo(False, "num"),
            "DeltaTriggerTime": FiltPerkeo(False, "num"),
            "DeltaCoinTime": FiltPerkeo(False, "num"),
            "EventNumber": FiltPerkeo(False, "num"),
        }

        self.pmt_data = None
        self.dptt = None
        self.stats = None
        self._pmt_thres = float(cnf["dataPerkeo"]["PMT_Thres"])
        self.hists = None
        self._hist_par = {
            "bin_count": int(cnf["dataPerkeo"]["ADC_hist_counts"]),
            "low_lim": int(cnf["dataPerkeo"]["ADC_hist_min"]),
            "up_lim": int(cnf["dataPerkeo"]["ADC_hist_max"]),
        }
        self.hist_sums = None
        self._histsum_par = {
            "bin_count": int(cnf["dataPerkeo"]["SUM_hist_counts"]),
            "low_lim": int(cnf["dataPerkeo"]["SUM_hist_min"]),
            "up_lim": int(cnf["dataPerkeo"]["SUM_hist_max"]),
        }

    def info(self, ball=False):
        """Print (Valid) Events, Cycles, content of branches/trees"""

        print("File name: \t" + self.filename)
        print(
            "Event Number: \t",
            self._ev_no,
            "\tCycle Number:\t",
            self._cy_no,
            "\t Mode: \t",
            self.mode,
        )
        print(
            "Valid Events: \t", self._ev_valid_no, "\tValid Cycles:\t", self.cy_valid_no
        )
        if ball:
            print("------------------- \t All Keys: \t -------------------")
            print(self.file.keys())
            print("------------------- \t dataTree \t -------------------")
            print(self.file["dataTree"].show())
            print("------------------- \t cycleTree \t -------------------")
            print(self.file["cycleTree"].show())

        return 0

    def print_filt(self):
        """Print current filter settings."""

        print("---\t cycleTree filters: \t ---")
        for i in self.cyclefilter:
            print(i, "\t\t", self.cyclefilter[i].active)
        print("---\t dataTree filters: \t ---")
        for i in self.datafilter:
            print(i, "\t\t", self.datafilter[i].active)

        return 0

    def calc_dptt(self):
        """Calculates DeltaPrevTriggerTime manually. Needs to be run before filter."""

        triggertime = self.file["dataTree"].array("TriggerTime")
        self.dptt = np.insert(
            (0.01 * (triggertime[1:] - triggertime[:-1])), 0, 666666666.0
        )  # [mys]
        return 0

    def calc_cyclefilt(self):
        """Given current filter settings, calc _cy_valid array."""

        print("---\t Applying cycleTree filters: \t ---")
        start_time = time.time()
        for i in self.cyclefilter:
            if self.cyclefilter[i].active:
                last_valcycles = self._cy_valid
                arr = self.file["cycleTree"].array(i)

                if self.cyclefilter[i].ftype == "bool":
                    self._cy_valid = arr == self.cyclefilter[i].rightval

                elif self.cyclefilter[i].ftype == "num":
                    self._cy_valid = (arr >= self.cyclefilter[i].lowerlimit) == (
                        arr < self.cyclefilter[i].upperlimit
                    )

                else:
                    print("ERROR: Invalid filter type.", " Filter cannot be applied.")
                    sys.exit(1)
                self._cy_valid = last_valcycles * self._cy_valid
        end_time = time.time()
        print(f"Time taken for cycle Filt: {end_time - start_time:0.5f} s")

        self.cy_valid_no = int(self._cy_valid.sum())
        print(
            "Remaining cycles:     ",
            self.cy_valid_no,
            "     from a total of:     ",
            self._cy_no,
        )
        assert self.cy_valid_no > 0, "ERROR: No cycles left. This would leave no data."

        if self.cy_valid_no == self._cy_no:
            print("All cycles valid. Deactivate data cycle filter")
            self.datafilter["Cycle"].active = False

        return 0

    def calc_datafilt(self):
        """Given current filter settings, calc _ev_valid array."""

        print("---\t Applying dataTree filters: \t ---")

        start_time = time.time()
        for i in self.datafilter:
            if self.datafilter[i].active:
                last_valevents = self._ev_valid
                if i == "DeltaPrevTriggerTime" and not self._bDPTT:
                    print("Doing DeltaPrevTriggerTime filter manually.")
                    arr = self.dptt
                else:
                    arr = self.file["dataTree"].array(i)

                if self.datafilter[i].ftype == "cyc":
                    print("Applying cycle filter in dataTree")

                    cycarr = self.file["cycleTree"].array("Cycle")

                    res = True
                    for k, cycval in enumerate(cycarr):
                        # check if cycle invalid
                        if self._cy_valid[k] == 0.0:
                            # locate bad cycles
                            int1 = arr == cycval
                            # invert array (now True is good and False is bad)
                            int2 = (int1 + 1) < 2
                            res *= int2
                    self._ev_valid = res

                elif self.datafilter[i].ftype == "bool":
                    print(
                        "Applying bool filter for:\t",
                        i,
                        "\t desired value:\t",
                        self.datafilter[i].rightval,
                    )
                    self._ev_valid = arr == self.datafilter[i].rightval

                elif self.datafilter[i].ftype == "num":
                    print("Applying number filter for:\t", i)
                    self._ev_valid = (arr >= self.datafilter[i].lowerlimit) == (
                        arr < self.datafilter[i].upperlimit
                    )

                else:
                    print(
                        "ERROR: Invalid filter type.",
                        "Filter cannot be applied.",
                    )
                    sys.exit(1)
                self._ev_valid = last_valevents * self._ev_valid
        end_time = time.time()
        print(f"Time taken for data Filt: {end_time - start_time:0.5f} s")
        self._ev_valid_no = int(self._ev_valid.sum())
        print(
            "Remaining events:     ",
            self._ev_valid_no,
            "     from a total of:     ",
            self._ev_no,
        )

        assert self._ev_valid_no > 0, "ERROR: No events left. This wouldn't leave data."

        return 0

    def calc_filt(self):
        """Calls filter calc. functions and determines valid events."""

        self._cy_valid = np.ones(self._cy_no)
        self.calc_cyclefilt()
        self._ev_valid = np.ones(self._ev_no)
        self.calc_datafilt()

        return 0

    def clear_filt(self):
        """Set all filters to inactive. Data needs to be refiltered."""

        print("---\t Clearing all filters: \t ---")
        for i in self.cyclefilter:
            self.cyclefilter[i].active = False
        for i in self.datafilter:
            self.datafilter[i].active = False

        return 0

    def set_filtdef(self):
        """Set filter default settings: filter invalid cycles only."""

        print("---\t Clearing all filters and set to default: \t ---")
        for i in self.cyclefilter:
            self.cyclefilter[i].active = False
        for i in self.datafilter:
            self.datafilter[i].active = False

        self.datafilter["Cycle"].active = True
        self.cyclefilter["Valid"].active = True
        self.cyclefilter["Valid"].rightval = True

        return 0

    def gen_pmtdata(self):
        """Generate pmt_data according to calc filter arrays."""

        data_pmt = self.file["dataTree"].array("PMT")
        data_pmtfilt = []
        for i, val in enumerate(data_pmt):
            if self._ev_valid[i] == 1:
                data_pmtfilt.append(val)

        data_pmt_tran = np.asarray(data_pmtfilt).transpose()
        no_tadc = len(data_pmt_tran)
        self._tadc = no_tadc - 1
        self.no_pmts = len(data_pmt_tran[0])
        # also set list length for self.hists
        self.hists = [None] * self.no_pmts
        self.hist_sums = [None] * 2

        # check for mode
        assert no_tadc != 0, "ERROR: TADC format is wrong."
        if no_tadc == 1:
            self.mode = 1  # Delta
        elif no_tadc == 2:
            self.mode = 2  # Both
        elif no_tadc == 33:
            self.mode = 3  # All
        assert self.mode is not None, "ERROR: Mode could not be determined"

        if self.mode == 1:
            self.pmt_data = data_pmt_tran[self._tadc]
        else:
            self.pmt_data = data_pmt_tran[self._tadc] - data_pmt_tran[0]

        return 0

    def gen_dptt(self):
        """Filter dptt data according to calculated filters."""

        dptt_data = self.dptt
        dptt_filt = []
        for i, val in enumerate(dptt_data):
            if self._ev_valid[i] == 1:
                dptt_filt.append(val)
        self.dptt = np.array(dptt_filt)

        return 0

    def ret_array_by_key(self, key: str) -> np.array:
        """Generate array from data according to calc filter arrays by keyword."""

        data_array = self.file["dataTree"].array(key)
        data_arrayfilt = []
        for i, val in enumerate(data_array):
            if self._ev_valid[i] == 1:
                data_arrayfilt.append(val)

        return np.asarray(data_arrayfilt)

    def calc_stats(self):
        """Calc. data stats. pmt_data needs to be generated first."""

        n_mean = np.zeros(self.no_pmts)
        n_sig = np.zeros(self.no_pmts)
        n_norm = np.zeros(self.no_pmts)
        n_sig2d = np.zeros([self.no_pmts, 2])
        n_norm2d = np.zeros([self.no_pmts, 2])
        n_pmt = np.zeros(self.no_pmts)

        for i in range(0, self.no_pmts):
            n_mean[i] = self.pmt_data[i].mean()
            n_sig[i] = self.pmt_data[i].std()
            n_sig2d[i][0] = self.pmt_data[i].std()
            n_norm[i] = self.pmt_data[i].max()
            n_norm2d[i][0] = self.pmt_data[i].max()
            if n_mean[i] > self._pmt_thres:
                n_pmt[i] = 1

        self.stats = {
            "mean": n_mean,
            "sig": n_sig,
            "maxval": n_norm,
            "active": n_pmt,
            "sig2D": n_sig2d,
            "norm2D": n_norm2d,
        }

        return 0

    def auto(self, setfiltmode: int = 0):
        """Run functions in correct order to get data out of file.

        This function calls all other necessary functions in
        appropriate order to generate pmt_data. DeltaPrevTriggerTime data is
        generated as well.

        Parameters
        ----------
        setfiltmode: int
            0 = use default settings (only filter invalid cycles/events)
            2 = ignore all filters and use raw data
            any other int = use set filters (before running this)
            (default 0)
        """

        # FIXME: First entry in dptt for both cases?
        if self._bDPTT:
            print("DPTT already exists.")  # [mus]
            self.dptt = 0.01 * self.file["dataTree"].array("DeltaPrevTriggerTime")
        else:
            print("DPTT doesnt exist. Is calculated.")
            self.calc_dptt()
        if setfiltmode == 0:
            self.set_filtdef()
        else:
            if setfiltmode == 2:
                self.clear_filt()
        self.calc_filt()
        self.gen_pmtdata()
        self.gen_dptt()
        self.calc_stats()

        return 0

    def ret_actpmt(self) -> list:
        """Returns list of PMT index of 'active' (see _pmt_thres) PMTs."""

        liste = []
        for i in range(0, len(self.stats["active"])):
            if self.stats["active"][i] == 1:
                liste.append(i)

        return liste

    def gen_hist(
        self, lpmt: list(int), cust_hist_par: list = None, cust_histsum_par: list = None
    ):
        """Generate histograms from pmt_data. Last function to call.

        Parameters
        ----------
        lpmt : list of int
            List of PMT index values. Could be ret_actpmt() return.
        """

        detsum0 = (self.pmt_data[:8]).sum(axis=0)
        detsum1 = (self.pmt_data[8:]).sum(axis=0)
        if cust_histsum_par is None:
            self.hist_sums[0] = HistPerkeo(detsum0, **self._histsum_par)
            self.hist_sums[1] = HistPerkeo(detsum1, **self._histsum_par)
        else:
            self.hist_sums[0] = HistPerkeo(detsum0, **cust_histsum_par)
            self.hist_sums[1] = HistPerkeo(detsum1, **cust_histsum_par)

        for i in lpmt:
            if cust_hist_par is None:
                self.hists[i] = HistPerkeo(self.pmt_data[i], **self._hist_par)
            else:
                self.hists[i] = HistPerkeo(self.pmt_data[i], **cust_hist_par)

        return 0
