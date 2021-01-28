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
        self.map = pd.DataFrame(columns=["time", "peak_list", "err_list", "gof"])
        self._calc()

    def _calc(self):
        """"""

        for i, meas in enumerate(filt_meas):
            if i in [61, 294, 625]:
                continue

            print(f"Meas No: {i}")
            time = meas.date_list[0]
            files = meas.file_list

            dataSrc, dataBg = [dP.RootPerkeo(files[0]), dP.RootPerkeo(files[1])]
            dataSrc.info()
            dataSrc.auto()
            dataBg.auto()
            dataSrc.gen_hist(list(range(0, dataSrc.no_pmts)))
            dataBg.gen_hist(list(range(0, dataBg.no_pmts)))

            for j, ival in enumerate(dataSrc.hists):
                ival.addhist(dataBg.hists[j], -dataSrc.cy_valid_no / dataBg.cy_valid_no)

            fitdataclass = eP.DoFitData(dataSrc, "DriftSn")
            fit_result = fitdataclass.fit()

            datepair = np.array([dataSrc.filedate, dataBg.filedate])
            pmt_np = np.array(list(map(lambda x: x[0], fit_result)))
            gof = np.array(list(map(lambda x: x[1], fit_result)))
            mu_val = np.array(list(map(lambda x: x[2].params["mu"].value, fit_result)))
            mu_err = np.array(list(map(lambda x: x[2].params["mu"].stderr, fit_result)))
            del dataSrc
            del dataBg

            meas_dict = {
                "time": time,
                "peak_list": mu_val,
                "err_list": mu_err,
                "gof": gof,
            }
            print(meas_dict)
            self.map = self.map.append(meas_dict, ignore_index=True)


pdm = PerkeoDriftMap(filt_meas)
dmap = pdm.map

file = dP.FilePerkeo("driftmap")
print("Write to file", file.dump(pdm))


fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Sn Drift study")

axs[0].set_title("Peak pos over time Det 0")
axs[1].set_title("Peak pos over time Det 1")
# axs[1].set_title("rChi2 over time")
axs.flat[0].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
axs.flat[1].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
# axs.flat[1].set(xlabel="Time [s]", ylabel="rChi2 [ ]")

peak_df = dmap["peak_list"].apply(pd.Series)
err_df = dmap["err_list"].apply(pd.Series)
rchi2_df = dmap["gof"].apply(pd.Series)

for PMT in range(8):

    axs[0].errorbar(
        dmap["time"], peak_df[PMT], yerr=err_df[PMT], fmt=".", label=f"PMT{PMT}"
    )
    axs[1].errorbar(
        dmap["time"],
        peak_df[PMT + 8],
        yerr=err_df[PMT + 8],
        fmt=".",
        label=f"PMT{PMT+8}",
    )

plt.show()

# corr_class = corrPerkeo(meas)
# corr_class.corrections["Pedestal"] = False
# corr_class.corrections["RateDepElec"] = False
# corr_class.corr(bstore=True, bwrite=False)
