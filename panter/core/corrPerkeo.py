"""Module for correcting Perkeo data."""

import configparser
import numpy as np
import dataPerkeo as dP
from dataloaderPerkeo import DLPerkeo

import subprocess
import uproot
import evalFunctions as eF
import evalPerkeo as eP
from panter.config.params import delt_pmt
from panter.config.params import k_pmt_fix

cnf = configparser.ConfigParser()
cnf.read("../config/evalRaw.ini")

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
    """"""

    def __init__(self, dataloader: DLPerkeo):
        self._dataloader = dataloader
        self._histpar_sum = SUM_hist_par
        self._beam_mtime = BEAM_MEAS_TIME
        self._corrections = {"Pedestal": True, "RateDepElec": False}

    def __calc_detsum(
        self, vals: list, start_it: int = 0
    ) -> [dP.HistPerkeo, dP.HistPerkeo]:
        """Calculate the detsum for list of ADC values."""

        detsum0 = np.array(vals[:8]).sum(axis=0)[start_it:]
        detsum1 = np.array(vals[8:]).sum(axis=0)[start_it:]
        hist0 = dP.HistPerkeo(detsum0, **self._histpar_sum)
        hist1 = dP.HistPerkeo(detsum1, **self._histpar_sum)

        return [hist0, hist1]

    def __set_corr(self):
        """Activate corrections from list"""
        pass

    def __filt_data(self, data: dP.RootPerkeo, bbeam=False, key=""):
        """"""

        data.info()
        if bbeam:
            data.set_filtdef()
            data.datafilter["DeltaTriggerTime"].active = True
            data.datafilter["DeltaTriggerTime"].upperlimit = self._beam_mtime[key][1]
            data.datafilter["DeltaTriggerTime"].lowerlimit = self._beam_mtime[key][0]
            data.auto(1)
        else:
            data.auto()

        return 0

    def __calc_corr(self, data: dP.RootPerkeo):
        """Calculate corrected amplitude for each event and file."""

        pedestals = [[0]] * data.no_pmts
        ampl_corr = [None] * data.no_pmts
        if self._corrections["Pedestal"]:
            pedestals = eP.PedPerkeo(data).ret_pedestals()
            # or get fixed values from params.py

        for i in range(0, data.no_pmts):
            ampl_corr[i] = data.pmt_data[i] - pedestals[i][0]

        if self._corrections["RateDepElec"]:
            # FIXME: Think about this [1:]!
            dptt = data.dptt[1:]
            for i in range(0, data.no_pmts):
                ampl_0 = ampl_corr[i][1:]
                ampl_1 = ampl_corr[i][:-1]
                ampl_corr[i] = eF.calc_Acorr_ratedep(
                    ampl_0, ampl_1, dptt, delta=delt_pmt[i], k=k_pmt_fix[i]
                )

        hist_old = self.__calc_detsum(data.pmt_data, 1)
        hist_new = self.__calc_detsum(ampl_corr)

        return [[hist_old, hist_new], data.cy_valid_no]

    def __corr_beam(self, ev_file: list):
        """"""

        res = []
        data_sg = dP.RootPerkeo(ev_file[0])
        data_bg = dP.RootPerkeo(ev_file[0])
        data_dict = {"sg": data_sg, "bg": data_bg}

        for (key, data) in data_dict.items():
            self.__filt_data(data, bbeam=True, key=key)
            r, s = self.__calc_corr(data)
            res.append(r)

        fac = (self._beam_mtime["sg"][1] - self._beam_mtime["sg"][0]) / (
            self._beam_mtime["bg"][1] - self._beam_mtime["bg"][0]
        )

        for det in [0, 1]:
            res[0][0][det].addhist(res[1][0][det], -fac)
            res[0][1][det].addhist(res[1][1][det], -fac)

        res_old = [res[0][0][0], res[0][0][1]]
        res_new = [res[0][1][0], res[0][1][1]]

        return [res_old, res_new]

    def __corr_src(self, ev_files: list):
        """"""

        res = []
        scal = []

        for file_name in ev_files:
            data = dP.RootPerkeo(file_name)
            self.__filt_data(data)

            r, s = self.__calc_corr(data)
            res.append(r)
            scal.append(s)

        for det in [0, 1]:
            res[0][0][det].addhist(res[1][0][det], -scal[0] / scal[1])
            res[0][1][det].addhist(res[1][1][det], -scal[0] / scal[1])

        res_old = [res[0][0][0], res[0][0][1]]
        res_new = [res[0][1][0], res[0][1][1]]

        return [res_old, res_new]

    def corr(self):
        """"""

        corr = ""
        for (corr_name, is_active) in self._corrections.items():
            if is_active:
                corr += corr_name

        cyc_no = 0
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
                [hist_o, hist_n] = self.__corr_beam(files)
            elif tp == 1:
                [hist_o, hist_n] = self.__corr_src(files)

            out_file_old = uproot.recreate(f"int_old.root")
            out_file_new = uproot.recreate(f"int_new.root")
            for det in [0, 1]:
                out_file_old[f"DetSum{det}"] = hist_o[det].ret_asnumpyhist()
                out_file_new[f"DetSum{det}"] = hist_n[det].ret_asnumpyhist()

            root_cmd = "/home/max/Software/root_install/bin/root"
            arg_old = f'recalcHistErr.cpp("int_old.root", "{src_name}_{cyc_no}_{corr}_old.root")'
            arg_new = f'recalcHistErr.cpp("int_new.root", "{src_name}_{cyc_no}_{corr}_new.root")'
            subprocess.run([root_cmd, arg_old])
            subprocess.run([root_cmd, arg_new])

            subprocess.run(["rm", "int_old.root"])
            subprocess.run(["rm", "int_new.root"])

        return 0
