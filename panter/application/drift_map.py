"""Calculate drift map from Sn measurements."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 3])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])


class PerkeoDriftMap:
    """"""

    def __init__(self, fmeas: np.array, bimp_pmt: bool = False, bimp_sn: bool = False):
        self._fmeas = fmeas
        self._outfile = ["sn_peak_map", "pmt_fac_map"]
        self.peak_wam = None
        if bimp_pmt:
            # try to import pmt factor map
            impfile = dP.FilePerkeo(self._outfile[1])
            self.pmt_map = impfile.imp()
            assert self.pmt_map.shape[0] > 0, "ERROR: PMT factor map empty."
        elif bimp_sn:
            # try to import sn peak map
            impfile = dP.FilePerkeo(self._outfile[0])
            self.sn_map, self.peak_wam = impfile.imp()
            assert self.sn_map.shape[0] > 0
            self.pmt_map = pd.DataFrame(columns=["time", "pmt_fac"])
            self._calc_pmt_fac()
        else:
            self.sn_map = pd.DataFrame(
                columns=["time", "peak_list", "err_list", "rchi2"]
            )
            self._calc_peak_pos()
            self.pmt_map = pd.DataFrame(columns=["time", "pmt_fac"])
            self._calc_pmt_fac()

    def _calc_peak_pos(self):
        """"""

        for i, meas in enumerate(self._fmeas):
            if i in [61, 294, 625]:
                continue

            print(f"Meas No: {i}")
            time = meas.date_list[0]
            if time > 2.5787e9:
                continue

            corr_class = corrPerkeo(dataloader=meas, mode=2)
            corr_class.corrections["DeadTime"] = True
            corr_class.corrections["Pedestal"] = True
            corr_class.corrections["RateDepElec"] = True
            corr_class.corr(bstore=True, bwrite=False)

            fitsettings = eFS.gaus_expmod
            fitsettings.plot_labels = [
                "SnSpec fit result",
                "ADC [ch]",
                "Counts [ ]",
            ]
            fitsettings.plotrange["x"] = [30.0, 4000.0]

            hists = corr_class.histograms[0][1]
            rchi2 = np.array([])
            mu_val = np.array([])
            mu_err = np.array([])

            for j in range(len(hists)):
                dofitclass = eP.DoFit(hists[j].hist)
                dofitclass.setup(fitsettings)
                dofitclass.set_bool("boutput", False)
                dofitclass.fit()

                fit_results = dofitclass.ret_results()
                rchi2 = np.append(rchi2, dofitclass.ret_gof()[0])
                mu_val = np.append(mu_val, fit_results.params["mu"].value)
                mu_err = np.append(mu_err, fit_results.params["mu"].stderr)

            meas_dict = {
                "time": time,
                "peak_list": mu_val,
                "err_list": mu_err,
                "rchi2": rchi2,
            }
            self.sn_map = self.sn_map.append(meas_dict, ignore_index=True)

        rchi2_df = self.sn_map["rchi2"].apply(pd.Series)
        peak_df = self.sn_map["peak_list"].apply(pd.Series)
        err_df = self.sn_map["err_list"].apply(pd.Series)

        rchi2_filter = rchi2_df < 1.5
        peak_df = peak_df[rchi2_filter]
        err_df = err_df[rchi2_filter]

        self.peak_wam = (peak_df / err_df ** 2).sum() / (1.0 / err_df ** 2).sum()

        outfile = dP.FilePerkeo(self._outfile[0])
        assert (
            outfile.dump([self.sn_map, self.peak_wam]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def _calc_pmt_fac(self):
        """"""

        for index, sn_meas in self.sn_map.iterrows():

            factors = (self.peak_wam / sn_meas["peak_list"]).to_numpy()
            print(factors)
            rchi2_filter = sn_meas["rchi2"] > 1.5
            factors[rchi2_filter] = None
            print(factors)
            pmt_dict = {
                "time": sn_meas["time"],
                "pmt_fac": factors,
            }

            self.pmt_map = self.pmt_map.append(pmt_dict, ignore_index=True)

        outfile = dP.FilePerkeo(self._outfile[1])
        assert outfile.dump(self.pmt_map) == 0, "ERROR: Export of drift map failed."

        return 0

    def plot_sn_map(self):
        """"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Sn Drift study")

        axs[0].set_title("Peak pos over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")

        peak_df = self.sn_map["peak_list"].apply(pd.Series)
        err_df = self.sn_map["err_list"].apply(pd.Series)

        for PMT in range(8):
            axs[0].errorbar(
                self.sn_map["time"],
                peak_df[PMT],
                yerr=err_df[PMT],
                fmt=".",
                label=f"PMT{PMT}",
            )
            axs[1].errorbar(
                self.sn_map["time"],
                peak_df[PMT + 8],
                yerr=err_df[PMT + 8],
                fmt=".",
                label=f"PMT{PMT + 8}",
            )
        plt.show()

        return 0

    def plot_pmt_map(self):
        """"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("PMT drift correction factors")

        axs[0].set_title("Factor over time Det 0")
        axs[1].set_title("Factor over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="Correction factor [ ]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="Correction factor [ ]")

        pmt_fac = self.pmt_map["pmt_fac"].apply(pd.Series)

        for PMT in range(8):
            axs[0].plot(
                self.pmt_map["time"],
                pmt_fac[PMT],
                ".",
                label=f"PMT{PMT}",
            )
            axs[1].plot(
                self.pmt_map["time"],
                pmt_fac[PMT + 8],
                ".",
                label=f"PMT{PMT + 8}",
            )
        plt.show()

        return 0


pdm = PerkeoDriftMap(filt_meas)
pdm.plot_sn_map()
pdm.plot_pmt_map()
