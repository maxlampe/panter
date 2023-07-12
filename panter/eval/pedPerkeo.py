"""Pedestal calculation class from data"""

import configparser

import matplotlib.pyplot as plt
import numpy as np

from panter.config import conf_path
from panter.config.evalFitSettings import gaus_simp
from panter.data.dataHistPerkeo import HistPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.eval.evalFit import DoFit

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")


class PedPerkeo:
    """
    Class for generating, plotting and storing pedestal values of PERKEO III datasets.

    Does not overwrite previous filters in dataclass RootPerkeo.

    Parameters
    ----------
    dataclass : RootPerkeo
    bplot_res, bplot_fit, bplot_log: False, False, False
        Activate plotting the pedestal results, plotting each fit result and plotting
        each fit result with a log scaled y-axis.
    bnaive_filt: False
        If True, deactivates standard filter (detector == 0/1) to get electronic Eigen-
        signal and fit a Gaussian to data below 500 channels for each PMT. Useful for
        e.g. electronics data with non-organic trigger patterns.
    range_detsum: None
        Filter DetSum for specific range (list). If None, it's not applied.
    range_dtt: None
        Filter DeltraTriggerTime for specific range (list). If None, it's not applied.

    Examples
    --------
    >>> data = RootPerkeo(filename)
    >>> pedtest = PedPerkeo(data)
    >>> pedtest.plot_pedestals()
    >>> print(pedtest.ret_pedestals())
    """

    def __init__(
        self,
        dataclass: RootPerkeo,
        bplot_res: bool = False,
        bplot_fit: bool = False,
        bplot_log: bool = False,
        bnaive_filt: bool = False,
        range_detsum: list = None,
        range_dtt: list = None,
        custom_hist_par: dict = None,
    ):
        self._dataclass = dataclass
        self._bplot_fit = bplot_fit
        self._bplot_log = bplot_log
        self._bnaive_filt = bnaive_filt
        self._range_detsum = range_detsum
        self._range_dtt = range_dtt

        if self._dataclass.no_pmts is None:
            self._dataclass.auto()
        if custom_hist_par is None:
            self._ped_hist_par = {
                "bin_count": int(cnf["dataPerkeo"]["PED_hist_counts"]),
                "low_lim": int(cnf["dataPerkeo"]["PED_hist_min"]),
                "up_lim": int(cnf["dataPerkeo"]["PED_hist_max"]),
            }
        else:
            self._ped_hist_par = custom_hist_par
        self._pedvalues = np.asarray(self.calc_ped())
        self.ped_hists = None
        if bplot_res:
            self.plot_pedestals()

    def calc_ped(self):
        """Calculating pedestals"""

        ped_list = [None] * self._dataclass.no_pmts
        ped_hists = np.asarray([None] * self._dataclass.no_pmts)

        for DET in [0, 1]:
            self._dataclass.clear_filt()
            if not self._bnaive_filt:
                self._dataclass.set_filt(
                    "data", fkey="Detector", active=True, ftype="bool", rightval=1 - DET
                )

            if self._range_detsum is not None:
                self._dataclass.set_filt(
                    "data",
                    fkey="DetSum",
                    active=True,
                    ftype="num",
                    low_lim=self._range_detsum[0],
                    up_lim=self._range_detsum[1],
                )
            if self._range_dtt is not None:
                self._dataclass.set_filt(
                    "data",
                    fkey="DeltaTriggerTime",
                    active=True,
                    ftype="num",
                    low_lim=self._range_dtt[0],
                    up_lim=self._range_dtt[1],
                )
            self._dataclass.auto(1)
            if DET == 0:
                for i in range(0, 8):
                    ped_hists[i] = HistPerkeo(
                        self._dataclass.pmt_data[i], **self._ped_hist_par
                    )

            elif DET == 1:
                for i in range(8, 16):
                    ped_hists[i] = HistPerkeo(
                        self._dataclass.pmt_data[i], **self._ped_hist_par
                    )

        for ind_hist, hist in enumerate(ped_hists):
            if hist is not None:
                histogram = hist.hist

                for i in [0, 1]:
                    fitclass = DoFit(histogram)
                    fitclass.setup(gaus_simp)
                    # FIXME: Why is this necessary?
                    # FIXME: Why does it change gaus_simp instead of fitclass att?
                    fitclass.set_bool("blimfit", False)
                    fitclass.set_bool("boutput", False)

                    if i == 0:
                        mu_init = histogram["x"][np.argmax(histogram["y"])]
                        fitclass.set_fitparam(namekey="mu", valpar=mu_init)
                        fitclass.set_fitparam(namekey="sig", valpar=30.0)
                        if self._bnaive_filt:
                            fitclass.limit_range([-np.inf, 500.0])
                        fitclass.fit()

                        if fitclass.ret_results() is not None:
                            mu_start = fitclass.ret_results().params["mu"].value
                            sig_start = np.abs(
                                fitclass.ret_results().params["sig"].value
                            )
                            fit_range = [
                                mu_start - 1.1 * sig_start,
                                mu_start + 0.5 * sig_start,
                            ]
                        else:
                            break

                    else:
                        fitclass.set_fitparam(namekey="mu", valpar=mu_start)
                        fitclass.set_fitparam(namekey="sig", valpar=sig_start)
                        fitclass.limit_range(fit_range)
                        if self._bplot_fit:
                            if self._bplot_log:
                                fitclass.blogscale = self._bplot_log
                                plot_yr = [0.1, histogram["y"].max() * 10.0]
                                fitclass.plotrange["y"] = plot_yr
                            fitclass.set_bool("boutput", True)
                            plot_xr = [
                                mu_start - 7.0 * sig_start,
                                mu_start + 7.0 * sig_start,
                            ]
                            fitclass.plotrange["x"] = plot_xr
                            fitclass.plot_labels = ["", "Energy [ch]", "Counts [ ]"]
                            # fitclass.plot_file = f"Ped{ind_hist}"
                            # fitclass.set_bool("bsave_fit", True)
                        fitclass.fit()

                if fitclass.ret_results() is not None:
                    ped_list[ind_hist] = [
                        fitclass.ret_results().params["mu"].value,
                        fitclass.ret_results().params["mu"].stderr,
                        np.abs(fitclass.ret_results().params["sig"].value),
                        fitclass.ret_results().params["sig"].stderr,
                        fitclass.ret_gof()[0],
                        fitclass.ret_gof()[1],
                    ]

        self.ped_hists = ped_hists
        self._dataclass.clear_filt()
        # TODO: there was an issue here. see pedestal_analysis
        # print(self.ped_hists)

        return ped_list

    def ret_pedestals(self) -> np.array(float):
        return self._pedvalues

    def plot_pedestals(self):
        """Plotting pedestal fit results."""

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

        axs[0].set_title("Pedestal Gaussian Mu w Err")
        axs.flat[0].set(ylabel="Ped Mu [ch]")
        axs[0].errorbar(
            x=range(0, 16),
            y=self._pedvalues.T[0],
            yerr=self._pedvalues.T[1],
            fmt=".",
        )

        axs[1].set_title("Pedestal Gaussian Sig w Err")
        axs.flat[1].set(ylabel="Ped Sig [ch]")
        axs[1].errorbar(
            x=range(0, 16),
            y=self._pedvalues.T[2],
            yerr=self._pedvalues.T[3],
            fmt=".",
        )
        plt.show()

        return 0


def main():
    fdir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    file = fdir + "data194348-70372_2.root"

    data = RootPerkeo(file)
    pedtest = PedPerkeo(data, bplot_fit=True, bplot_log=True)
    pedtest.plot_pedestals()


if __name__ == "__main__":
    main()
