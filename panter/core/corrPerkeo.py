"""Module for correcting PERKEO data."""

import configparser
import numpy as np
import pandas as pd
import subprocess
import copy
import uproot
import panter.core.dataPerkeo as dP
import panter.core.evalFunctions as eF
import panter.core.evalPerkeo as eP
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core import core_path
from panter.config import conf_path
from panter.application import appl_path
from panter.config.params import delt_pmt
from panter.config.params import k_pmt_fix

cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

PMT_hist_par = {
    "bin_count": int(cnf["dataPerkeo"]["ADC_hist_counts"]),
    "low_lim": int(cnf["dataPerkeo"]["ADC_hist_min"]),
    "up_lim": int(cnf["dataPerkeo"]["ADC_hist_max"]),
}

SUM_hist_par = {
    "bin_count": int(cnf["dataPerkeo"]["SUM_hist_counts"]),
    "low_lim": int(cnf["dataPerkeo"]["SUM_hist_min"]),
    "up_lim": int(cnf["dataPerkeo"]["SUM_hist_max"]),
}
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


class CorrPerkeo:
    """Class for doing correction on PERKEO data.

    Takes a data loader and corrects all entries in it.

    Parameters
    ----------
    dataloader: DLPerkeo
    mode: {0, 1, 2}
        Mode variable to determine calculated spectra.
        O = total DetSum (only one spectra is returned)
        1 = only Sum over each detector PMTs (two spectra are returned)
        2 = all PMTs will be treated individually (no_pmt spectra)
    bonlynew: bool
        Only create corrected spectra instead of uncorrected spectra as well.
    ped_arr: np.array
        Array of pedestal values to be used instead of calculated from data file.
    weight_arr: np.array
        Array of individual PMT weights to multiply each event for each PMT.

    Attributes
    ----------
    corrections : {
        "Pedestal": True,
        "RateDepElec": False,
        "DeadTime": True,
        "Drift": True
    }
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
    Can be used to just calculate background subtracted data. If corrections were to be
    set to True, data would be individually corrected and then background subtracted.
    Corrections can be all turned off or on with the shown class method.

    >>> meas = DLPerkeo().ret_meas()
    >>> corr_class = CorrPerkeo(meas)
    >>> corr_class.set_all_corr(bactive=False)
    >>> corr_class.corrections["Pedestal"] = True
    >>> corr_class.corrections["Drift"] = True
    >>> corr_class.corr(bstore=True, bwrite=False)
    """

    def __init__(
        self,
        dataloader: DLPerkeo,
        mode: int = 0,
        bonlynew: bool = True,
        ped_arr=None,
        weight_arr=None,
    ):
        self._dataloader = dataloader
        self._bonlynew = bonlynew
        self._ped_arr = ped_arr
        self._weight_arr = weight_arr
        self._mode = mode
        self._histpar_sum = SUM_hist_par
        self._histpar_pmt = PMT_hist_par
        self._beam_mtime = BEAM_MEAS_TIME
        self.corrections = {
            "Pedestal": True,
            "RateDepElec": False,
            "DeadTime": True,
            "Drift": True,
        }
        self.histograms = []
        self.hist_concat = None
        self.addition_filters = []

        self._drift_map = None

    def _calc_detsum(
        self, vals: list, start_it: int = 0
    ) -> [dP.HistPerkeo, dP.HistPerkeo]:
        """Calculate the DetSum for list of ADC values."""

        calc_hists = []
        if self._mode == 0:
            det_sum_tot = np.array(vals[:]).sum(axis=0)[start_it:]
            calc_hists.append(dP.HistPerkeo(det_sum_tot, **self._histpar_sum))
        elif self._mode == 1:
            det_sum_0 = np.array(vals[:8]).sum(axis=0)[start_it:]
            det_sum_1 = np.array(vals[8:]).sum(axis=0)[start_it:]

            calc_hists.append(dP.HistPerkeo(det_sum_0, **self._histpar_sum))
            calc_hists.append(dP.HistPerkeo(det_sum_1, **self._histpar_sum))

        elif self._mode == 2:
            for val in vals:
                calc_hists.append(dP.HistPerkeo(val, **self._histpar_pmt))

        return calc_hists

    def _set_corr(self):
        """Activate corrections from list."""
        pass

    def set_all_corr(self, bactive: bool):
        """Switch all corrections to active or inactive"""

        for corr in self.corrections:
            self.corrections[corr] = bactive

        return 0

    def clear(self):
        """Clear relevant attributes enabling re-usability without new instantiation."""

        self.addition_filters = []
        self.histograms = []

        return 0

    def _filt_data(self, data: dP.RootPerkeo, bbeam=False, key=""):
        """Filter data set."""

        data.info()
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
            data.auto(1)
        else:
            if len(self.addition_filters) != 0:
                data.set_filtdef()
                for entry in self.addition_filters:
                    data.set_filt(**entry)
                data.auto(1)
            else:
                data.auto()

        return 0

    def _calc_corr(self, data: dP.RootPerkeo):
        """Calculate corrected amplitude for each event and file."""

        pedestals = [[0]] * data.no_pmts
        ampl_corr = [None] * data.no_pmts
        drift_factors = [1.0] * data.no_pmts
        binvalid = False

        if self.corrections["Drift"]:
            impfile = dP.FilePerkeo(conf_path + "/pmt_fac_map.p")
            self._drift_map, _ = impfile.imp()

            time_stamps = self._drift_map["time"]
            curr_time = data.filedate
            diff_time = np.abs(time_stamps - curr_time)
            nearest_drift = diff_time.idxmin()

            if diff_time[nearest_drift] < 7200.0:
                print("ERROR: Last drift meas more than 2h away")
                binvalid = True

            drift_factors = self._drift_map["pmt_fac"][nearest_drift]

        if self.corrections["Pedestal"]:
            if self._ped_arr is None:
                datacop = copy.copy(data)
                datacop.set_filtdef()
                pedestals = eP.PedPerkeo(datacop).ret_pedestals()
            else:
                print("Warning: Using passed pedestals.")
                pedestals = self._ped_arr

        if self._weight_arr is None:
            self._weight_arr = np.ones(data.no_pmts)

        for i in range(0, data.no_pmts):
            if pedestals[i] is not None:
                ampl_corr[i] = (
                    (data.pmt_data[i] - pedestals[i][0])
                    * drift_factors[i]
                    * self._weight_arr[i]
                )
            else:
                ampl_corr[i] = None

        if self.corrections["RateDepElec"]:
            # FIXME: Think about this [1:]!
            dptt = data.dptt[1:]
            for i in range(0, data.no_pmts):
                ampl_0 = ampl_corr[i][1:]
                ampl_1 = ampl_corr[i][:-1]
                ampl_corr[i] = eF.calc_acorr_ratedep(
                    ampl_0, ampl_1, dptt, delta=delt_pmt[i], k=k_pmt_fix[i]
                )

        if self._bonlynew:
            hist_old = None
        else:
            hist_old = self._calc_detsum(data.pmt_data)
        hist_new = self._calc_detsum(ampl_corr)

        if self.corrections["DeadTime"]:
            for hist in range(len(hist_new)):
                if not self._bonlynew:
                    hist_old[hist].scal(data.dt_fac)
                hist_new[hist].scal(data.dt_fac)

        return [[hist_old, hist_new], data.cy_valid_no, binvalid]

    def _corr_nobg(self, ev_file: list):
        """Correct measurement without background subtraction"""

        res = []
        data_sg = dP.RootPerkeo(ev_file[0])

        self._filt_data(data_sg)
        r, s, i = self._calc_corr(data_sg)
        if i:
            return [None, None]
        res.append(r)

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def _corr_beam(self, ev_file: list):
        """Correct beam-like data (i.e. background in same file)."""

        res = []
        data_sg = dP.RootPerkeo(ev_file[0])
        data_bg = dP.RootPerkeo(ev_file[0])
        data_dict = {"sg": data_sg, "bg": data_bg}

        for (key, data) in data_dict.items():
            self._filt_data(data, bbeam=True, key=key)
            r, s, i = self._calc_corr(data)
            if i:
                return [None, None]
            res.append(r)

        fac = [
            (self._beam_mtime["sg"][1] - self._beam_mtime["sg"][0])
            / (self._beam_mtime["bg"][1] - self._beam_mtime["bg"][0])
        ] * 2

        for hist_no in range(len(res[0][1])):
            if not self._bonlynew:
                res[0][0][hist_no].addhist(res[1][0][hist_no], -fac[0])
            res[0][1][hist_no].addhist(res[1][1][hist_no], -fac[1])

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def _corr_src(self, ev_files: list):
        """Correct source-like data (i.e. background in different file)."""

        res = []
        scal = []

        for file_name in ev_files:
            data = dP.RootPerkeo(file_name)
            self._filt_data(data)

            r, s, i = self._calc_corr(data)
            if i:
                return [None, None]
            res.append(r)
            scal.append(s)

        for hist_no in range(len(res[0][1])):
            if not self._bonlynew:
                res[0][0][hist_no].addhist(res[1][0][hist_no], -scal[0] / scal[1])
            res[0][1][hist_no].addhist(res[1][1][hist_no], -scal[0] / scal[1])

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def corr(self, bstore: bool = False, bwrite: bool = True, bconcat: bool = False):
        """Correcting data according to chosen settings.

        Parameters
        ----------
        bstore: False
            Bool whether to append created histograms in self.histograms
        bwrite: True
            Bool whether to write created histograms to a ROOT file.
        bconcat
            Bool to concatenate spectra.
        """

        if not bwrite and not bstore:
            print("WARNING: Doing nothing with data ")

        corr = ""
        for (corr_name, is_active) in self.corrections.items():
            if is_active:
                corr += corr_name

        cyc_no = 0

        # Catch single MeasPerkeo inputs
        try:
            self._dataloader.shape
        except AttributeError:
            self._dataloader = [self._dataloader]

        for meas in self._dataloader:
            tp = meas.tp
            src = meas.src
            files = meas.file_list
            if meas.cyc_no is not None:
                cyc_no = meas.cyc_no
            else:
                cyc_no += 1

            src_name = "PerkeoHist"
            if src == 5:
                src_name = "Beam"
            if (np.array([0, 1, 2, 3, 4]) == src).sum() < 0:
                src_name = f"Src{src}"

            if tp == 0:
                [hist_o, hist_n] = self._corr_beam(files)
            elif tp == 1:
                [hist_o, hist_n] = self._corr_src(files)
            elif tp == 2:
                [hist_o, hist_n] = self._corr_nobg(files)

            if bwrite:
                filename = f"{src_name}_{cyc_no}_{corr}.root"
                if self._mode == 0:
                    hist_n[0].write2root(histname=f"DetSumTot", filename=filename)
                    if not self._bonlynew:
                        hist_o[0].write2root(
                            histname=f"DetSumTot", filename=filename, bupdate=True
                        )

                elif self._mode == 1:
                    det = 0
                    hist_n[det].write2root(histname=f"DetSum{det}", filename=filename)
                    if not self._bonlynew:
                        hist_o[det].write2root(
                            histname=f"DetSum{det}", filename=filename, bupdate=True
                        )
                    det = 1
                    hist_n[det].write2root(
                        histname=f"DetSum{det}", filename=filename, bupdate=True
                    )
                    if not self._bonlynew:
                        hist_o[det].write2root(
                            histname=f"DetSum{det}", filename=filename, bupdate=True
                        )

                elif self._mode == 2:
                    det = 0
                    hist_n[det].write2root(histname=f"DetSumTot", filename=filename)
                    if not self._bonlynew:
                        hist_o[det].write2root(
                            histname=f"DetSumTot", filename=filename, bupdate=True
                        )
                    for det in range(1, 16):
                        hist_n[det].write2root(
                            histname=f"DetSumTot", filename=filename, bupdate=True
                        )
                        if not self._bonlynew:
                            hist_o[det].write2root(
                                histname=f"DetSumTot", filename=filename, bupdate=True
                            )

            if bconcat:
                # TODO: Implement concatenation for all modes
                if self._mode == 0:
                    if self.hist_concat is None:
                        if hist_n is not None:
                            self.hist_concat = hist_n[0]
                    if self.hist_concat is not None:
                        if hist_n is not None:
                            self.hist_concat.addhist(hist_n[0])

            if bstore:
                self.histograms.append(np.asarray([hist_o, hist_n]))

        self.histograms = np.asarray(self.histograms)

        return 0
