"""Module for correcting PERKEO data."""

import configparser
import numpy as np
import subprocess
import copy
import uproot
import panter.core.dataPerkeo as dP
import panter.core.evalFunctions as eF
import panter.core.evalPerkeo as eP
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core import core_path
from panter.config import conf_path
from panter.config.params import delt_pmt
from panter.config.params import k_pmt_fix

cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

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


class corrPerkeo:
    """Class for doing correction on PERKEO data.

    Takes a data loader and corrects all entries in it.

    Parameters
    ----------
    dataloader: DLPerkeo

    Attributes
    ----------
    corrections : {"Pedestal": True, "RateDepElec": False}
    histograms : []
        List to store created histograms to, if bstore=True in self.corr()
    addition_filters: []
        List of individual entries to be used as filters with data
        RootPerkeo.set_filt() in _filt_data().

    Examples
    --------
    Can be used to just calculate background subtracted data. If corrections were to be
    set to True, data would be individually corrected and then background subtracted.

    >>> meas = DLPerkeo()
    >>> corr_class = corrPerkeo(meas)
    >>> corr_class.corrections["Pedestal"] = False
    >>> corr_class.corrections["RateDepElec"] = False
    >>> corr_class.corr(bstore=True, bwrite=False)
    """

    def __init__(self, dataloader: DLPerkeo):
        self._dataloader = dataloader
        self._histpar_sum = SUM_hist_par
        self._beam_mtime = BEAM_MEAS_TIME
        self.corrections = {"Pedestal": True, "RateDepElec": False}
        self.histograms = []
        # TODO merge dead time into corrections
        self.corr_deadtime = True
        self.addition_filters = []

    def _calc_detsum(
        self, vals: list, start_it: int = 0
    ) -> [dP.HistPerkeo, dP.HistPerkeo]:
        """Calculate the DetSum for list of ADC values."""

        detsum0 = np.array(vals[:8]).sum(axis=0)[start_it:]
        detsum1 = np.array(vals[8:]).sum(axis=0)[start_it:]
        hist0 = dP.HistPerkeo(detsum0, **self._histpar_sum)
        hist1 = dP.HistPerkeo(detsum1, **self._histpar_sum)

        return [hist0, hist1]

    def _set_corr(self):
        """Activate corrections from list."""
        pass

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

        if self.corrections["Pedestal"]:
            datacop = copy.copy(data)
            datacop.set_filtdef()
            pedestals = eP.PedPerkeo(datacop).ret_pedestals()
            # or get fixed values from params.py

        for i in range(0, data.no_pmts):
            ampl_corr[i] = data.pmt_data[i] - pedestals[i][0]

        if self.corrections["RateDepElec"]:
            # FIXME: Think about this [1:]!
            dptt = data.dptt[1:]
            for i in range(0, data.no_pmts):
                ampl_0 = ampl_corr[i][1:]
                ampl_1 = ampl_corr[i][:-1]
                ampl_corr[i] = eF.calc_Acorr_ratedep(
                    ampl_0, ampl_1, dptt, delta=delt_pmt[i], k=k_pmt_fix[i]
                )

        hist_old = self._calc_detsum(data.pmt_data)
        hist_new = self._calc_detsum(ampl_corr)

        if self.corr_deadtime:
            for det in [0, 1]:
                hist_old[det].scal(data.dt_fac[det])
                hist_new[det].scal(data.dt_fac[det])

        return [[hist_old, hist_new], data.cy_valid_no]

    def _corr_beam(self, ev_file: list):
        """Correct beam-like data (i.e. background in same file)."""

        res = []
        data_sg = dP.RootPerkeo(ev_file[0])
        data_bg = dP.RootPerkeo(ev_file[0])
        data_dict = {"sg": data_sg, "bg": data_bg}

        for (key, data) in data_dict.items():
            self._filt_data(data, bbeam=True, key=key)
            r, s = self._calc_corr(data)
            res.append(r)

        fac = [
            (self._beam_mtime["sg"][1] - self._beam_mtime["sg"][0])
            / (self._beam_mtime["bg"][1] - self._beam_mtime["bg"][0])
        ] * 2

        for det in [0, 1]:
            res[0][0][det].addhist(res[1][0][det], -fac[0])
            res[0][1][det].addhist(res[1][1][det], -fac[1])

        res_old = [res[0][0][0], res[0][0][1]]
        res_new = [res[0][1][0], res[0][1][1]]

        return [res_old, res_new]

    def _corr_src(self, ev_files: list):
        """Correct source-like data (i.e. background in different file)."""

        res = []
        scal = []

        for file_name in ev_files:
            data = dP.RootPerkeo(file_name)
            self._filt_data(data)

            r, s = self._calc_corr(data)
            res.append(r)
            scal.append(s)

        for det in [0, 1]:
            res[0][0][det].addhist(res[1][0][det], -scal[0] / scal[1])
            res[0][1][det].addhist(res[1][1][det], -scal[0] / scal[1])

        res_old = [res[0][0][0], res[0][0][1]]
        res_new = [res[0][1][0], res[0][1][1]]

        return [res_old, res_new]

    def corr(self, bstore: bool = False, bwrite: bool = True):
        """Correcting data according to chosen settings.

        Parameters
        ----------
         bstore: False
            Bool whether to append created histograms in self.histograms
         bwrite: True
            Bool whether to write created histograms to a ROOT file.
        """

        if not bwrite and not bstore:
            print("WARNING: Doing nothing with data ")

        corr = ""
        for (corr_name, is_active) in self.corrections.items():
            if is_active:
                corr += corr_name

        cyc_no = 0

        if type(self._dataloader) != "panter.core.dataloaderPerkeo.DLPerkeo":
            batches = [self._dataloader]
        else:
            batches = self._dataloader
        for meas in batches:
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

            if bwrite:
                out_file_old = uproot.recreate(f"int_old.root")
                out_file_new = uproot.recreate(f"int_new.root")
                for det in [0, 1]:
                    out_file_old[f"DetSum{det}"] = hist_o[det].ret_asnumpyhist()
                    out_file_new[f"DetSum{det}"] = hist_n[det].ret_asnumpyhist()

                root_cmd = "/home/max/Software/root_install/bin/root"
                arg_old = (
                    f"{core_path}/recalcHistErr.cpp"
                    + f'("int_old.root", "{src_name}_{cyc_no}_{corr}_old.root")'
                )
                arg_new = (
                    f"{core_path}/recalcHistErr.cpp"
                    + f'("int_new.root", "{src_name}_{cyc_no}_{corr}_new.root")'
                )
                subprocess.run([root_cmd, arg_old])
                subprocess.run([root_cmd, arg_new])

                subprocess.run(["rm", "int_old.root"])
                subprocess.run(["rm", "int_new.root"])

            if bstore:
                self.histograms.append(np.asarray([hist_o, hist_n]))
        self.histograms = np.asarray(self.histograms)

        return 0
