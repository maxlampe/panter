"""Calculate simple trigger map over beam files."""

import datetime
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panter.base.mapPerkeo import MapPerkeo
from panter.config import conf_path

from panter.config.evalFitSettings import trigger_func
from panter.data.dataMeasPerkeo import MeasPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.data.dataMisc import FilePerkeo
from panter.eval.evalFit import DoFit

output_path = os.getcwd()


class TriggerMap(MapPerkeo):
    """"""

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_fitres: bool = True,
    ):
        super().__init__(fmeas=fmeas, level=1, bimport=[bimp_fitres])
        self._outfile = ["trigger_fitres_map.p"]

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
                    "a_0",
                    "a_0_err",
                    "p_0",
                    "p_0_err",
                    "rChi2_0",
                    "a_1",
                    "a_1_err",
                    "p_1",
                    "p_1_err",
                    "rChi2_1",
                ]
            )
            self._calc_fits()

            return False

    def _calc_fits(self, n_files_sum: int = 5):
        """Do fits of trigger time of flight data."""

        hist_trigger = [None] * 2

        for i in range(0, self._fmeas.shape[0], n_files_sum):
            print(f"Meas No: {i}")
            if i % 25 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile[0])

            time = 0.0

            for primary_detector in [0, 1]:
                master_both = None
                master_onlysec = None
                det_prim = primary_detector
                for j in range(n_files_sum):
                    meas = self._fmeas[i + j]
                    time += meas.date_list[0]
                    hist_both, hist_onlysec = self.trigger_raw(meas, det_prim)

                    if j == 0:
                        master_onlysec = hist_onlysec[det_prim]
                        master_both = hist_both[det_prim]
                    else:
                        master_onlysec.addhist(hist_onlysec[det_prim])
                        master_both.addhist(hist_both[det_prim])

                master_both.divbyhist(master_onlysec)
                hist_trigger[det_prim] = master_both

            time = time / (2 * n_files_sum)

            fit_results = [None] * 2
            rChi2 = [None] * 2
            for j in range(len(hist_trigger)):
                fit_class = DoFit(hist_trigger[j].hist)
                fit_class.setup(trigger_func)
                fit_class.limit_range([500, 15e3])
                fit_class.set_fitparam(namekey="a", valpar=0.003)
                fit_class.set_fitparam(namekey="p", valpar=0.655)
                fit_class.set_bool("boutput", True)
                fit_class.plotrange["x"] = [0, 15e3]
                fit_class.plotrange["y"] = [-0.2, 1.2]
                fit_class.plot_labels = ["Trigger Det", "ADC [ch]", "Trigger prob. [ ]"]
                fit_class.fit()

                fit_results[j] = fit_class.ret_results()
                rChi2[j] = fit_class.ret_gof()[0]

            meas_dict = {
                "time": time,
                "a_0": fit_results[0].params["a"].value,
                "a_0_err": fit_results[0].params["a"].stderr,
                "p_0": fit_results[0].params["p"].value,
                "p_0_err": fit_results[0].params["p"].stderr,
                "rChi2_0": rChi2[0],
                "a_1": fit_results[1].params["a"].value,
                "a_1_err": fit_results[1].params["a"].stderr,
                "p_1": fit_results[1].params["p"].value,
                "p_1_err": fit_results[1].params["p"].stderr,
                "rChi2_1": rChi2[1],
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile[0]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def plot_fit_results(self, bsave: bool = False):
        """Plot fit results"""

        fig, axs = plt.subplots(
            3,
            2,
            sharex=True,
            figsize=(16, 12),
        )
        fig.subplots_adjust(hspace=0)
        fig.suptitle("Trigger Fit Results over time")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        for i in range(6):
            axs.flat[i].xaxis.set_major_formatter(xfmt)

        keys = [
            "a_0",
            "a_0_err",
            "a_1",
            "a_1_err",
            "p_0",
            "p_0_err",
            "p_1",
            "p_1_err",
            "rChi2_0",
            "rChi2_1",
        ]
        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

        for i in range(0, int((len(keys)) * 0.5) - 1):
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

        for i in range(2):
            axs.flat[4 + i].plot(
                dates_plot,
                self.maps[0][f"rChi2_{i}"].apply(pd.Series)[0],
                ".",
                label=f"rChi2_{i}",
            )
            axs.flat[4 + i].set(xlabel="Time [ ]", ylabel=f"rChi2_{i} [ ]")

        for i in range(6):
            axs.flat[i].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[0][:-1] + "png", dpi=300)
        plt.show()

        return 0

    @staticmethod
    def trigger_raw(meas: MeasPerkeo, det_main: int):
        """Calculate trigger function for one detector from raw data."""

        det_bac = 1 - det_main
        data = RootPerkeo(meas.file_list[0])
        data.set_filtdef()
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_bac,
        )
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=4e9,
            up_lim=4e10,
            index=det_main,
        )
        data.auto(1)
        data.gen_hist([])
        hist_onlybac = data.hist_sums

        data.set_filtdef()
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_bac,
        )
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_main,
        )
        data.set_filt(
            "data", fkey="Detector", active=True, ftype="bool", rightval=det_main
        )
        data.auto(1)
        data.gen_hist([])

        hist_b = data.hist_sums
        hist_onlybac[det_main].addhist(hist_b[det_main])

        return [hist_b, hist_onlybac]


def main():
    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(dir)
    dataloader.auto()
    # filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 69536])
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])

    trigger_map = TriggerMap(fmeas=filt_meas, bimp_fitres=True)
    trigger_map()
    trigger_map.plot_fit_results()


if __name__ == "__main__":
    main()
