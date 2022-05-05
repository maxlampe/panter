"""Module for correcting PERKEO data."""

import configparser
import copy

import numpy as np

from panter.base.corrBase import CorrBase
from panter.config import conf_path
from panter.config.params import delt_pmt
from panter.config.params import k_pmt_fix
from panter.data.dataHistPerkeo import HistPerkeo
from panter.data.dataMisc import FilePerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.eval.evalFunctions import calc_acorr_ratedep
from panter.eval.pedPerkeo import PedPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo


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


class CorrPerkeo(CorrBase):
    """Class for doing correction on PERKEO data.

    Takes a data loader and corrects all entries in it.

    Parameters
    ----------
    dataloader: np.array()
    mode: {0, 1, 2}
        Mode variable to determine calculated spectra.
        O = total DetSum (only one spectra is returned)
        1 = only Sum over each detector PMTs (two spectra are returned)
        2 = all PMTs will be treated individually (no_pmt spectra)
    bonlynew: True
        Only create corrected spectra instead of uncorrected spectra as well.
    bdetsum_drift: True
        Use DetSum drift correction instead of individual PMT factors.
    ped_arr: np.array
        Array of pedestal values to be used instead of calculated from data file.
    bgped_arr: np.array
        Array of pedestal values to be used for bg data (works only for MeasP type 1).
    weight_arr: np.array
        Array of individual PMT weights to multiply each event for each PMT.
    pmt_sum_selection: list
        List of pmt indices for which pmts should be summed up in spectra creation.
        Requires mode=0, as everything else would make no sense.
    custom_sum_hist_par: dict
        Custom histogram parameters for DetSum histograms.
    custom_pmt_hist_par: dict
        Custom histogram parameters individual PMT histograms.

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
        dataloader: np.array,
        mode: int = 0,
        bonlynew: bool = True,
        bdetsum_drift: bool = True,
        ped_arr=None,
        bgped_arr=None,
        weight_arr=None,
        pmt_sum_selection=None,
        custom_sum_hist_par=None,
        custom_pmt_hist_par=None,
    ):
        super().__init__(dataloader=dataloader, bonlynew=bonlynew)
        self._bdetsum_drift = bdetsum_drift
        self._ped_arr = ped_arr
        self._bgped_arr = bgped_arr
        self._weight_arr = weight_arr
        self._mode = mode
        self._pmt_sum_selection = pmt_sum_selection

        if self._pmt_sum_selection is not None:
            assert self._mode == 0, "ERROR: Wrong mode for custom pmt sum selection."

        if custom_sum_hist_par is None:
            self._histpar_sum = SUM_hist_par
        else:
            self._histpar_sum = custom_sum_hist_par
        if custom_pmt_hist_par is None:
            self._histpar_pmt = PMT_hist_par
        else:
            self._histpar_pmt = custom_pmt_hist_par

        # TODO: Add check for valid corrections when setting them
        self.corrections = {
            "Pedestal": True,
            "RateDepElec": True,
            "DeadTime": True,
            "Drift": True,
            "Scan2D": True,
        }
        self._drift_map = None
        self._drift_gprs = [None] * 2
        self._scan2d_map = None

    def _calc_detsum(
        self, vals: np.array, start_it: int = 0
    ) -> [HistPerkeo, HistPerkeo]:
        """Calculate the DetSum for list of ADC values."""

        calc_hists = []
        if self._mode == 0:
            if self._pmt_sum_selection is None:
                det_sum_tot = np.array(vals[:]).sum(axis=0)[start_it:]
            else:
                det_sum_tot = np.array(vals[self._pmt_sum_selection]).sum(axis=0)[
                    start_it:
                ]
            calc_hists.append(HistPerkeo(det_sum_tot, **self._histpar_sum))
        elif self._mode == 1:
            det_sum_0 = np.array(vals[:8]).sum(axis=0)[start_it:]
            det_sum_1 = np.array(vals[8:]).sum(axis=0)[start_it:]

            calc_hists.append(HistPerkeo(det_sum_0, **self._histpar_sum))
            calc_hists.append(HistPerkeo(det_sum_1, **self._histpar_sum))

        elif self._mode == 2:
            for val in vals:
                calc_hists.append(HistPerkeo(val, **self._histpar_pmt))

        return calc_hists

    def _calc_corr(self, data: RootPerkeo, buse_bgped: bool = False):
        """Calculate corrected amplitude for each event and file."""

        pedestals = [[0]] * data.no_pmts
        ampl_corr = [None] * data.no_pmts
        drift_factors = np.ones(data.no_pmts)
        scan2d_factors = np.ones(data.no_pmts)
        binvalid = False

        if self.corrections["Scan2D"]:
            impfile = FilePerkeo(conf_path + "/det_fac_2D_map.p")
            self._scan2d_map, _ = impfile.imp()

            time_stamps = self._scan2d_map["time"]
            diff_time = np.abs(time_stamps - data.filedate)
            nearest_2d = diff_time.idxmin()

            # if diff_time[nearest_2d] < 7200.0:
            #     print("ERROR: Last meas more than 2h away")
            #     binvalid = True

            scan2d_factors = self._scan2d_map["pmt_fac"][nearest_2d]

        if self.corrections["Drift"]:
            assert (
                self._bdetsum_drift
            ), "Error: Only drift det sum correction implemented at the moment."

            # FIXME: do not do this here for each correction
            from panter.eval.evalDriftGPR import GPRDrift
            import torch

            for det in [0, 1]:
                gpr_class = GPRDrift(detector=det)
                gpr_class.load_model(file_name=f"{conf_path}/gpr_model_det{det}.plk")
                self._drift_gprs[det] = gpr_class

            time_stamp = data.filedate
            time_stamp = self._drift_gprs[0].dataclass.timestamp_to_data(time_stamp)

            det_sum_fac = [
                self._drift_gprs[0](torch.tensor(time_stamp))[0],
                self._drift_gprs[1](torch.tensor(time_stamp))[0],
            ]

            drift_factors = np.asarray(
                [[det_sum_fac[0]] * 8, [det_sum_fac[1]] * 8]
            ).flatten()

        if self.corrections["Pedestal"]:
            if self._ped_arr is None:
                datacop = copy.copy(data)
                datacop.set_filtdef()
                pedestals = PedPerkeo(datacop).ret_pedestals()
            else:
                if buse_bgped:
                    pedestals = self._bgped_arr
                else:
                    pedestals = self._ped_arr

        if self._weight_arr is None:
            self._weight_arr = np.ones(data.no_pmts)

        for i in range(0, data.no_pmts):
            if pedestals[i] is not None:
                ampl_corr[i] = data.pmt_data[i] - pedestals[i][0]
            else:
                ampl_corr[i] = None

        if self.corrections["RateDepElec"]:
            # FIXME: Think about this [1:]!
            dptt = data.dptt[1:]
            for i in range(0, data.no_pmts):
                ampl_0 = ampl_corr[i][1:]
                ampl_1 = ampl_corr[i][:-1]
                ampl_corr[i] = calc_acorr_ratedep(
                    ampl_0, ampl_1, dptt, delta=delt_pmt[i], k=k_pmt_fix[i]
                )

        for i in range(0, data.no_pmts):
            ampl_corr[i] = (
                ampl_corr[i]
                * drift_factors[i]
                * scan2d_factors[i]
                * self._weight_arr[i]
            )

        ampl_corr = np.asarray(ampl_corr)
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
        data_sg = RootPerkeo(ev_file[0])

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
        data_sg = RootPerkeo(ev_file[0])
        data_bg = RootPerkeo(ev_file[0])
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

        for ind, file_name in enumerate(ev_files):
            data = RootPerkeo(file_name)
            self._filt_data(data)

            if ind == 1:
                r, s, i = self._calc_corr(data, buse_bgped=True)
            else:
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
            if (np.array([0, 1, 2, 3, 4]) == src).sum() > 0:
                src_name = f"Src{src}"

            if tp == 0:
                [hist_o, hist_n] = self._corr_beam(files)
            elif tp == 1:
                [hist_o, hist_n] = self._corr_src(files)
            elif tp == 2:
                [hist_o, hist_n] = self._corr_nobg(files)

            if bwrite:
                filename = f"{src_name}_{cyc_no}_{corr}"
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
                else:
                    raise NotImplementedError(
                        f"Concatenation not implemented for mode {self._mode}"
                    )

            if bstore:
                self.histograms.append(np.asarray([hist_o, hist_n]))

        self.histograms = np.asarray(self.histograms)

        return 0


def main():
    data_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(data_dir)
    dataloader.auto()
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])[-120:-80]

    corr_class = CorrPerkeo(filt_meas, mode=0)
    corr_class.set_all_corr(bactive=False)
    corr_class.corrections["Drift"] = True
    corr_class.corrections["Scan2D"] = True
    corr_class.corrections["RateDepElec"] = True
    corr_class.corrections["Pedestal"] = True
    corr_class.corrections["DeadTime"] = True

    corr_class.addition_filters.append(
        {
            "tree": "data",
            "fkey": "Detector",
            "active": True,
            "ftype": "bool",
            "rightval": 0,
        }
    )

    corr_class.corr(bstore=True, bwrite=False, bconcat=True)

    histp = corr_class.histograms[0]
    test = histp[1][0]

    test.plot_hist(
        title="Beta spectrum",
        xlabel="Energy [ch]",
        ylabel="Counts [ ]",
        # rng=[0.0, 40e3, -30.0, 2200.0],
    )
    print(test.stats)


if __name__ == "__main__":
    main()
