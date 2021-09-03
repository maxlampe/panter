""""""

from __future__ import annotations

import configparser
import os
import sys
import time

import numpy as np
import uproot

from panter.config import conf_path
from panter.data.dataHistPerkeo import HistPerkeo
from panter.data.dataMisc import FiltPerkeo

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")


class RootPerkeo:
    """Class for top layer PERKEO root file management.

    Takes filename and can generate data histograms according to set
    filters. Currently only for all PMTs individually. Has several helper functions.

    Parameters
    ----------
    filename : str
    bverbose : False
        En-/Disables general outputs when handling data with RootPerkeo.

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
    val_rtime : float
        Valid measurement time. After cycles are filtered, measurement time is summed up
        for remaining cycles. None on instantiation.
    dt_fac : float
        Dead time correction factor to correct number of detected events as a scaling
        factor. n_true = n_meas * dt_fac. None on instantiation.
    deadtime : float
        Dead time value from measurement. Imported from ini file (evalRaw.ini).
    stats : dict of lists
        Dictionary for PMT statistics: mean, stddev, max xval, 'active'
        for each PMT (array in pmt_data). None on instantiation.
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
            data.set_filt(
                "data",
                fkey="DeltaTriggerTime",
                active=True,
                ftype="num",
                low_lim=x,
                up_lim=y,
            )
            data.auto(1)
    >>> data.gen_hist(data.ret_actpmt())
    >>> dtt = 0.01 * data.ret_array_by_key("DeltaTriggerTime") # [mus]
    """

    def __init__(self, filename: str, bverbose: bool = False):
        self.filename = filename
        self.bverbose = bverbose

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

        self.cyclefilter = []
        self.datafilter = []

        self.pmt_data = None
        self.dptt = None
        self.val_rtime = None
        self.chop_freq = None
        self.dt_fac = None
        self.deadtime = float(cnf["dataPerkeo"]["DeadTime"])
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
        for filt in self.cyclefilter:
            print(filt.fkey, "\t\t", filt.active)
        print("---\t dataTree filters: \t ---")
        for filt in self.datafilter:
            print(filt.fkey, "\t\t", filt.active)

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

        if self.bverbose:
            print("---\t Applying cycleTree filters: \t ---")
        start_time = time.time()
        for filt in self.cyclefilter:
            if filt.active:
                last_valcycles = self._cy_valid
                arr = self.file["cycleTree"].array(filt.fkey)

                if filt.ftype == "bool":
                    self._cy_valid = arr == filt.rightval

                elif filt.ftype == "num":
                    self._cy_valid = (arr >= filt.low_lim) == (arr < filt.up_lim)

                else:
                    print("ERROR: Invalid filter type.", " Filter cannot be applied.")
                    sys.exit(1)
                self._cy_valid = last_valcycles * self._cy_valid
        end_time = time.time()
        if self.bverbose:
            print(f"Time taken for cycle Filt: {end_time - start_time:0.5f} s")

        self.cy_valid_no = int(self._cy_valid.sum())
        if self.bverbose:
            print(
                "Remaining cycles:     ",
                self.cy_valid_no,
                "     from a total of:     ",
                self._cy_no,
            )
        assert self.cy_valid_no > 0, "ERROR: No cycles left. This would leave no data."

        if self.cy_valid_no == self._cy_no:
            if self.bverbose:
                print("All cycles valid. Deactivate data cycle filter")
            for filt in self.datafilter:
                if filt.fkey == "Cycle":
                    filt.active = False

        return 0

    def calc_datafilt(self):
        """Given current filter settings, calc _ev_valid array."""

        if self.bverbose:
            print("---\t Applying dataTree filters: \t ---")

        start_time = time.time()
        for filt in self.datafilter:
            if filt.active:
                last_valevents = self._ev_valid
                if filt.fkey == "DeltaPrevTriggerTime" and not self._bDPTT:
                    if self.bverbose:
                        print("Doing DeltaPrevTriggerTime filter manually.")
                    arr = self.dptt
                else:
                    arr = self.file["dataTree"].array(filt.fkey)

                if filt.ftype == "cyc":
                    if self.bverbose:
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

                elif filt.ftype == "bool":
                    if self.bverbose:
                        print(
                            "Applying bool filter for:\t",
                            filt.fkey,
                            "\t desired value:\t",
                            filt.rightval,
                        )
                    self._ev_valid = arr == filt.rightval

                elif filt.ftype == "num":
                    if self.bverbose:
                        print("Applying number filter for:\t", filt.fkey)
                    if filt.index is not None:
                        arr = arr[:, filt.index]
                    self._ev_valid = (arr >= filt.low_lim) == (arr < filt.up_lim)

                else:
                    print(
                        "ERROR: Invalid filter type.",
                        "Filter cannot be applied.",
                    )
                    sys.exit(1)
                # TODO: why is this necessary?
                try:
                    self._ev_valid = last_valevents * self._ev_valid
                except MemoryError:
                    for ind, entry in enumerate(last_valevents):
                        self._ev_valid[ind] = float(self._ev_valid[ind] * entry)
                    self._ev_valid = self._ev_valid.flatten()

        end_time = time.time()
        if self.bverbose:
            print(f"Time taken for data Filt: {end_time - start_time:0.5f} s")
        self._ev_valid_no = int(self._ev_valid.sum())
        if self.bverbose:
            print(
                "Remaining events:     ",
                self._ev_valid_no,
                "     from a total of:     ",
                self._ev_no,
            )

        # FIXME!
        # assert self._ev_valid_no > 0, "ERROR: No events left. This wouldn't leave data."

        return 0

    def calc_filt(self):
        """Calls filter calc. functions and determines valid events."""

        self._cy_valid = np.ones(self._cy_no)
        self.calc_cyclefilt()
        self._ev_valid = np.ones(self._ev_no)
        self.calc_datafilt()

        return 0

    def set_filt(self, tree: str, **kwargs):
        """Set filter"""

        if tree == "data":
            self.datafilter.append(FiltPerkeo(**kwargs))
        if tree == "cycle":
            self.cyclefilter.append(FiltPerkeo(**kwargs))

        return 0

    def clear_filt(self):
        """Set all filters to inactive. Data needs to be refiltered."""

        if self.bverbose:
            print("---\t Clearing all filters: \t ---")
        self.cyclefilter.clear()
        self.datafilter.clear()

        return 0

    def set_filtdef(self):
        """Set filter default settings: filter invalid cycles only."""

        if self.bverbose:
            print("---\t Clearing all filters and set to default: \t ---")
        self.clear_filt()
        self.set_filt("cycle", fkey="Valid", active=True, ftype="bool", rightval=True)
        self.set_filt("data", fkey="Cycle", active=True, ftype="cyc")

        return 0

    def calc_times(self):
        """Calculate total measurement time and dead time correction factor.

        Uses formula for non-paralyzable analysis. Prints corrected total rate."""

        self.val_rtime = (
            self.file["cycleTree"].array("RealTime") * self._cy_valid
        ).sum()

        self.val_rtime = self.val_rtime / 1e8

        # need correction factor for burst rate, not average rate. check filters
        burst_rate = 1.0

        dt_error = "ERROR: Set filter probably breaks dead time validity."
        dtt_filter_count = 0
        for filt_perkeo in self.datafilter:
            assert filt_perkeo.fkey != "TriggerTime", dt_error
            assert filt_perkeo.fkey != "ChopperTime", dt_error
            assert filt_perkeo.fkey != "DeltaCoinTime", dt_error

            if filt_perkeo.fkey == "DeltaTriggerTime":
                dtt_filter_count += 1

                self.chop_freq = (
                    self.file["cycleTree"].array("ChopperSpeed") * self._cy_valid
                ).sum() / self.cy_valid_no

                delta_dtt = (filt_perkeo.up_lim - filt_perkeo.low_lim) / 1e8
                burst_rate *= delta_dtt * self.chop_freq
        assert dtt_filter_count <= 1, dt_error

        dead_time = self.deadtime * self._ev_valid_no
        self.dt_fac = 1.0 / (1.0 - dead_time / (self.val_rtime * burst_rate))

        return 0

    def gen_pmtdata(self):
        """Generate pmt_data according to calc filter arrays."""

        data_pmt = self.file["dataTree"].array("PMT")
        self._ev_valid = np.asarray(self._ev_valid, dtype=bool)
        data_pmtfilt = data_pmt[self._ev_valid]

        data_pmt_tran = np.asarray(data_pmtfilt).transpose()
        no_tadc = len(data_pmt_tran)
        self._tadc = no_tadc - 1
        self.no_pmts = len(data_pmt_tran[0])
        # also set list length for self.hists
        self.hists = [None] * self.no_pmts
        self.hist_sums = [None] * 3

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
            if self.pmt_data[i].shape[0] != 0:
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

    def auto(self, set_mode: int = 0):
        """Run functions in correct order to get PMT data out of file.

        This function calls all other necessary functions in
        appropriate order to generate pmt_data. DeltaPrevTriggerTime data is
        generated as well.

        Parameters
        ----------
        set_mode: int
            0 = use default settings (only filter invalid cycles/events)
            2 = ignore all filters and use raw data
            any other int = use set filters (before running this)
            (default 0)
        """

        # FIXME: First entry in dptt for both cases?
        if self._bDPTT:
            if self.bverbose:
                print("DPTT already exists.")  # [mus]
            self.dptt = 0.01 * self.file["dataTree"].array("DeltaPrevTriggerTime")
        else:
            if self.bverbose:
                print("DPTT doesnt exist. Is calculated.")
            self.calc_dptt()
        if set_mode == 0:
            self.set_filtdef()
        else:
            if set_mode == 2:
                self.clear_filt()
        self.calc_filt()
        self.calc_times()
        self.gen_pmtdata()
        self.gen_dptt()
        self.calc_stats()

    def ret_actpmt(self) -> list:
        """Returns list of PMT index of 'active' (see _pmt_thres) PMTs."""

        liste = []
        for i in range(0, len(self.stats["active"])):
            if self.stats["active"][i] == 1:
                liste.append(i)

        return liste

    def gen_hist(
        self, lpmt: list, cust_hist_par: dict = None, cust_histsum_par: dict = None
    ):
        """Generate histograms from pmt_data. Last function to call.

        Parameters
        ----------
        lpmt : list of int
            List of PMT index values. Could be ret_actpmt() return.
        cust_hist_par
            Custom histogram parameters for PMT spectra
        cust_histsum_par
            Custom histogram parameters for DetSum (only over one detector) spectra
        """

        detsum0 = (self.pmt_data[:8]).sum(axis=0)
        detsum1 = (self.pmt_data[8:]).sum(axis=0)
        detsum_all = self.pmt_data.sum(axis=0)
        if cust_histsum_par is None:
            self.hist_sums[0] = HistPerkeo(detsum0, **self._histsum_par)
            self.hist_sums[1] = HistPerkeo(detsum1, **self._histsum_par)
            self.hist_sums[2] = HistPerkeo(detsum_all, **self._histsum_par)
        else:
            self.hist_sums[0] = HistPerkeo(detsum0, **cust_histsum_par)
            self.hist_sums[1] = HistPerkeo(detsum1, **cust_histsum_par)
            self.hist_sums[2] = HistPerkeo(detsum_all, **cust_histsum_par)

        for i in lpmt:
            if cust_hist_par is None:
                self.hists[i] = HistPerkeo(self.pmt_data[i], **self._hist_par)
            else:
                self.hists[i] = HistPerkeo(self.pmt_data[i], **cust_hist_par)

        return 0
