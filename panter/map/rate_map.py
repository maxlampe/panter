"""Calculate simple uncorrected rate map over beam files."""

import datetime
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from panter.base.mapPerkeo import MapPerkeo
from panter.config import conf_path
from panter.data.dataMeasPerkeo import MeasPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.data.dataMisc import FilePerkeo

output_path = os.getcwd()


class RateMap(MapPerkeo):
    """"""

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_fitres: bool = True,
    ):
        super().__init__(fmeas=fmeas, level=1, bimport=[bimp_fitres])
        self._outfile = ["rate_map.p"]

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""

        if level == 0 and bimp:
            # try to import fit result map
            impfile = FilePerkeo(f"{conf_path}/{self._outfile[0]}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[level].shape[0] > 0, "ERROR: Result map empty."

            return True
        else:
            self.maps[0] = pd.DataFrame(
                columns=[
                    "time",
                    "r_0",
                    "r_0_err",
                    "r_1",
                    "r_1_err",
                    "pmt_rates"
                ]
            )
            self._calc_fits()

            return False

    def _calc_fits(self):
        """Extract event counts over time from data."""

        for i, meas in enumerate(self._fmeas):
            print(f"Meas No: {i} - {meas.cyc_no}")
            if i % 10 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile[0])

            time = meas.date_list[0]

            # extract rate from raw data file
            # hist_new[hist].scal(data.dt_fac)
            rates, pmt_rates = self.calc_event_rate(meas)
            pmt_rates = np.array(pmt_rates)

            meas_dict = {
                "time": time,
                "r_0": rates[0],
                "r_0_err": rates[1],
                "r_1": rates[2],
                "r_1_err": rates[3],
                "pmt_rates": pmt_rates
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile[0]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def plot_fit_results(self, bsave: bool = False):
        """Plot rates"""

        fig, axs = plt.subplots(
            2,
            2,
            sharex=True,
            figsize=(16, 12),
        )
        fig.subplots_adjust(hspace=0)
        fig.suptitle("Event rates over time")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        for i in range(4):
            axs.flat[i].xaxis.set_major_formatter(xfmt)

        keys = [
            "r_0",
            "r_0_err",
            "r_1",
            "r_1_err",
        ]
        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

        for i in range(2):
            vals = self.maps[0][keys[2 * i]].apply(pd.Series)
            vals_err = self.maps[0][keys[2 * i + 1]].apply(pd.Series)

            axs.flat[i].errorbar(
                dates_plot,
                vals[0],
                yerr=vals_err[0],
                fmt=".",
                label=f"{keys[2 * i]}",
            )
            axs.flat[i].set(xlabel="Time [ ]", ylabel=f"detector {keys[2 * i]} [ ]")
            axs.flat[i].legend()
        axs.flat[0].set_ylim([1150.0, 1520.0])
        axs.flat[1].set_ylim([1870.0, 2250.0])

        for i in range(2):
            vals = self.maps[0]["pmt_rates"].apply(pd.Series)

            for j in range(8):
                smoothed = savgol_filter(vals[8 * i + j], 15, 3)
                axs.flat[i + 2].plot(
                    dates_plot,
                    smoothed,
                    "-",
                    label=f"PMT{8 * i + j}",
                )
                # axs.flat[i + 2].plot(
                #     dates_plot,
                #     vals[8 * i + j],
                #     ".",
                #     label=f"PMT{8 * i + j}",
                # )
            axs.flat[i + 2].set(xlabel="Time [ ]", ylabel=f"SMOOTHED PMT_hist integral / (all_events * time) [ ]")
            axs.flat[i + 2].legend()

        if bsave:
            plt.savefig(output_path + "/" + self._outfile[0][:-1] + "png", dpi=300)
        plt.show()

        return 0

    @staticmethod
    def calc_event_rate(meas: MeasPerkeo):
        """Calculate event rate from data."""

        data = RootPerkeo(meas.file_list[0])
        rates = [None] * 2 * 2
        pmt_rates = [None] * 16
        for det in [0, 1]:
            data.set_filtdef()
            data.set_filt(
                "data", fkey="Detector", active=True, ftype="bool", rightval=det
            )
            data.auto(1)
            data.gen_hist(list(range(data.no_pmts)))
            n_events = data.hist_sums[2].stats["noevents"]
            rates[det * 2] = n_events / data.val_rtime
            rates[det * 2 + 1] = np.sqrt(n_events) / data.val_rtime
            for j in range(8):
                pmt_rates[8 * det + j] = data.hists[8 * det + j].integral / (
                    data.val_rtime * n_events
                )

            # TODO: data.dt_fac ? use dead time correction here?

        return rates, pmt_rates


def main():
    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(dir)
    dataloader.auto()
    # filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 69536])
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])

    rate_map = RateMap(fmeas=filt_meas, bimp_fitres=True)
    rate_map()
    rate_map.plot_fit_results()


if __name__ == "__main__":
    main()
