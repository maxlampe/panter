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

    def __init__(self, fmeas: MeasPerkeo):
        self.fmeas = fmeas
        self.outfile = "driftmap"
        self.map = pd.DataFrame(columns=["time", "peak_list", "err_list", "gof"])
        if self.map.shape[0] <= 0:
            self._calc()

    def _calc(self):
        """"""

        for i, meas in enumerate(filt_meas):
            if i in [61, 294, 625]:
                continue

            print(f"Meas No: {i}")
            time = meas.date_list[0]
            if time > 1.5788e9:
                continue

            corr_class = corrPerkeo(dataloader=meas, mode=2)
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
            fit_result = []
            for j in range(len(hists)):
                dofitclass = eP.DoFit(hists[j].hist)
                dofitclass.setup(fitsettings)
                dofitclass.set_bool("boutput", False)
                dofitclass.fit()

                fit_result.append([j, dofitclass.ret_gof(), dofitclass.ret_results()])

            gof = np.array(list(map(lambda x: x[1][0], fit_result)))
            mu_val = np.array(list(map(lambda x: x[2].params["mu"].value, fit_result)))
            mu_err = np.array(list(map(lambda x: x[2].params["mu"].stderr, fit_result)))

            meas_dict = {
                "time": time,
                "peak_list": mu_val,
                "err_list": mu_err,
                "gof": gof,
            }
            print(meas_dict)
            self.map = self.map.append(meas_dict, ignore_index=True)

        outfile = dP.FilePerkeo(self.outfile)
        assert outfile.dump(self.map) == 0, "ERROR: Export of drift map failed."

        return 0

    def plot_map(self):
        """"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Sn Drift study")

        axs[0].set_title("Peak pos over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")

        peak_df = self.map["peak_list"].apply(pd.Series)
        err_df = self.map["err_list"].apply(pd.Series)
        rchi2_df = self.map["gof"].apply(pd.Series)

        for PMT in range(8):
            axs[0].errorbar(
                self.map["time"],
                peak_df[PMT],
                yerr=err_df[PMT],
                fmt=".",
                label=f"PMT{PMT}",
            )
            axs[1].errorbar(
                self.map["time"],
                peak_df[PMT + 8],
                yerr=err_df[PMT + 8],
                fmt=".",
                label=f"PMT{PMT + 8}",
            )
        plt.show()

        return 0


pdm = PerkeoDriftMap(filt_meas)
pdm.plot_map()
