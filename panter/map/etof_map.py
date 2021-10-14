"""Calculate corrected E-ToF map with skewed gaussian fits"""

import datetime
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panter.base.corrSimple import CorrSimple
from panter.base.mapPerkeo import MapPerkeo
from panter.config import conf_path
from panter.config.evalFitSettings import etof_peaks
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.data.dataMisc import FilePerkeo
from panter.eval.evalFit import DoFit

output_path = os.getcwd()
plt.rcParams.update({"font.size": 12})


class EToFMap(MapPerkeo):
    """"""

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_fitres: bool = True,
    ):
        super().__init__(fmeas=fmeas, level=1, bimport=[bimp_fitres])
        self._outfile = ["etof_fitres_map.p"]

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""

        if level == 0 and bimp:
            # try to import fit result map
            impfile = FilePerkeo(f"{conf_path}/{self._outfile[0]}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[level].shape[0] > 0, "ERROR: Fit result map empty."

            return True
        else:
            self.maps[0] = pd.DataFrame(
                columns=[
                    "time",
                    "a",
                    "a_err",
                    "loc",
                    "loc_err",
                    "scale",
                    "scale_err",
                    "shift",
                    "shift_err",
                    "const_bg",
                    "const_bg_err",
                    "norm_p",
                    "norm_p_err",
                    "norm_n",
                    "norm_n_err",
                    "rChi2",
                ]
            )
            self._calc_fits()

            return False

    def _calc_fits(self):
        """Do fits of electron time of flight data."""

        for i, meas in enumerate(self._fmeas):
            print(f"Meas No: {i} - {meas.cyc_no}")
            if i % 25 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile[0])

            time = meas.date_list[0]

            corr_class = CorrSimple(
                dataloader=meas,
                branch_key="CoinTime",
                hist_par={"bin_count": 200, "low_lim": -20, "up_lim": 20},
            )

            corr_class.addition_filters.append(
                {
                    "tree": "data",
                    "fkey": "CoinTime",
                    "active": True,
                    "ftype": "num",
                    "low_lim": 0,
                    "up_lim": 2000000000,
                    "index": 0,
                }
            )
            corr_class.addition_filters.append(
                {
                    "tree": "data",
                    "fkey": "CoinTime",
                    "active": True,
                    "ftype": "num",
                    "low_lim": 0,
                    "up_lim": 2000000000,
                    "index": 1,
                }
            )
            corr_class.corr(bstore=True, bwrite=False)

            fitsettings = etof_peaks
            fitsettings.plot_labels = [
                "E-ToF - corrected",
                "DiffCoinTime [ch]",
                "Counts [ ]",
            ]
            # fitsettings.plotrange["x"] = [30.0, 16000.0]

            hist = corr_class.histograms[0][1]
            hist_data = hist[0].hist
            fitclass = DoFit(hist_data)
            fitclass.setup(fitsettings)
            fitclass.set_bool("boutput", True)
            fitclass.limit_range([-9.0, 9.0])
            fitclass.fit()
            fit_results = fitclass.ret_results()

            meas_dict = {
                "time": time,
                "a": fit_results.params["a"].value,
                "a_err": fit_results.params["a"].stderr,
                "loc": fit_results.params["loc"].value,
                "loc_err": fit_results.params["loc"].stderr,
                "scale": fit_results.params["scale"].value,
                "scale_err": fit_results.params["scale"].stderr,
                "shift": fit_results.params["shift"].value,
                "shift_err": fit_results.params["shift"].stderr,
                "const_bg": fit_results.params["const_bg"].value,
                "const_bg_err": fit_results.params["const_bg"].stderr,
                "norm_p": fit_results.params["norm_p"].value,
                "norm_p_err": fit_results.params["norm_p"].stderr,
                "norm_n": fit_results.params["norm_n"].value,
                "norm_n_err": fit_results.params["norm_n"].stderr,
                "rChi2": fitclass.ret_gof()[0],
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile[0]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def plot_fit_results(self, bsave: bool = False):
        """Plot fit results"""

        fig, axs = plt.subplots(
            4,
            2,
            sharex=True,
            figsize=(16, 12),
            # gridspec_kw={"height_ratios": [6, 2]},
        )
        fig.subplots_adjust(hspace=0)
        fig.suptitle("EToF Fit Results over time")

        # axs[0].set_title("2D Factor over time Det 0")
        # axs[1].set_title("2D Factor over time Det 1")
        # axs.flat[0].set(xlabel="Time [s]", ylabel="2D Correction factor [ ]")
        # axs.flat[1].set(xlabel="Time [s]", ylabel="2D Correction factor [ ]")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        for i in range(8):
            axs.flat[i].xaxis.set_major_formatter(xfmt)

        keys = [
            "a",
            "a_err",
            "loc",
            "loc_err",
            "scale",
            "scale_err",
            "shift",
            "shift_err",
            "const_bg",
            "const_bg_err",
            "norm_p",
            "norm_p_err",
            "norm_n",
            "norm_n_err",
            "rChi2",
        ]
        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

        for i in range(0, int((len(keys)) * 0.5)):
            vals = self.maps[0][keys[2 * i]].apply(pd.Series)
            vals_err = self.maps[0][keys[2 * i + 1]].apply(pd.Series)

            axs.flat[i].errorbar(
                dates_plot,
                vals[0],
                yerr=vals_err[0],
                fmt=".",
                label=f"{keys[2 * i]}",
            )
            axs.flat[i].set(xlabel="Time [ ]", ylabel=f"{keys[2 * i]} [ ]")

        axs.flat[7].plot(
            dates_plot,
            self.maps[0]["rChi2"].apply(pd.Series)[0],
            ".",
            label="rChi2",
        )
        axs.flat[7].set(xlabel="Time [ ]", ylabel="rChi2 [ ]")

        for i in range(8):
            axs.flat[i].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[0][:-1] + "png", dpi=300)
        plt.show()

        return 0


def main():
    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(dir)
    dataloader.auto()
    # filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 69536])
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])

    etofmap = EToFMap(fmeas=filt_meas, bimp_fitres=True)
    etofmap()
    etofmap.plot_fit_results()


if __name__ == "__main__":
    main()
