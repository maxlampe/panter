""""""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np

from panter.base.histBase import HistBase
from panter.data.dataMisc import ret_hist2d

from panter.data.dataloaderPerkeo import DLPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.eval.corrPerkeo import CorrPerkeo

output_path = os.getcwd()
plt.rcParams.update({"font.size": 12})


class Hist2DPerkeo(HistBase):
    """"""

    def __init__(
        self,
        data: np.array,
        bin_count=None,
        low_lim=None,
        up_lim=None,
    ):
        super().__init__(data=data)
        if up_lim is None:
            up_lim = [52000.0, 52000.0]
        if low_lim is None:
            low_lim = [0.0, 0.0]
        if bin_count is None:
            bin_count = [1024, 1024]
        for dim in [0, 1]:
            assert low_lim[dim] <= up_lim[dim], "Error: lower limit > than upper limit."

        if self._data.shape != ():
            self.n_events = self._data.shape[1]
            self.mean = [self._data[0].mean(), self._data[1].mean()]
            self.stdv = [self._data[0].std(), self._data[1].std()]
            self.stats = {
                "mean": self.mean,
                "std": self.stdv,
                "noevents": self.n_events,
            }
        self.parameters = {"bin_count": bin_count, "low_lim": low_lim, "up_lim": up_lim}
        self.bin_widths = [(up_lim[i] - low_lim[i]) / bin_count[i] for i in [0, 1]]

        if self._data.shape != ():
            self.hist = ret_hist2d(self._data, **self.parameters)
        else:
            self.hist = None

        self.integral = (self.hist["z"] * self.bin_widths[0] * self.bin_widths[1]).sum()

    def _calc_stats(self):
        """Calculate mean and biased variance of histogram based on bin content."""

        self.n_events = self.hist["z"].sum()
        # FIXME: Mean and variance need to be calculate from bin content!
        # self.mean = (self.hist["x"] * self.hist["y"]).sum() / self.n_events
        # var = ((self.hist["x"] - self.mean) ** 2 * self.hist["y"]).sum()/self.n_events
        # self.stdv = np.sqrt(var)

        self.stats = {
            "mean": self.mean,
            "std": self.stdv,
            "noevents": self.n_events,
        }
        self.integral = (self.hist["z"] * self.bin_widths[0] * self.bin_widths[1]).sum()

    def plot_hist(
        self,
        rng: list = None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        zlabel: str = None,
        bsavefig: bool = False,
        bxlog: bool = False,
        bylog: bool = False,
        filename: str = None,
        vlims: list = None,
        fsize: int = None,
        zticks: list = None,
        xticks: list = None,
        yticks: list = None,
        zcut: float = None,
    ):
        """Plot histogram."""

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        mesh_x, mesh_y = np.meshgrid(self.hist["x_edges"], self.hist["y_edges"])
        if vlims is None:
            vlims = [None] * 2
        z_vals = self.hist["z"]
        if zcut is not None:
            z_vals[z_vals <= zcut] = None
        ims = axs.pcolormesh(
            mesh_x,
            mesh_y,
            z_vals,
            cmap="plasma",
            vmin=vlims[0],
            vmax=vlims[1],
            linewidth=0,
            rasterized=True,
        )
        # ims.set_edgecolor('face')
        cbar = fig.colorbar(
            ims,
            ticks=zticks,
        )
        cbar.set_label(label=zlabel, size=fsize)
        plt.title(title, fontsize=fsize)
        plt.ylabel(ylabel, fontsize=fsize)
        plt.xlabel(xlabel, fontsize=fsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        if fsize is not None:
            plt.tick_params(labelsize=fsize)
            cbar.ax.tick_params(labelsize=fsize)

        if rng is not None:
            plt.axis(rng)
        if bylog:
            plt.yscale("log")
        if bxlog:
            plt.xscale("log")

        if bsavefig:
            if filename is None:
                filename = "hist2dperkeo"
            plt.savefig(f"{output_path}/{filename}.pdf", dpi=300)
        plt.show()

    def addhist(self, hist_p: Hist2DPerkeo, fac: float = 1.0):
        """Add another histogram to existing one with multiplicand."""

        assert self.parameters == hist_p.parameters, "ERROR: Binning does not match."

        new_hist = {
            **self.hist,
            "z": self.hist["z"] + fac * hist_p.hist["z"],
            "err": np.sqrt(self.hist["err"] ** 2 + (fac * hist_p.hist["err"]) ** 2),
        }
        # Changes input ret_hist like in Root
        self.hist = new_hist
        self._calc_stats()

    def scal(self, fac: float):
        """Scale histogram by a factor."""

        new_hist = {
            **self.hist,
            "z": (self.hist["z"] * fac),
            "err": np.sqrt((fac * self.hist["err"]) ** 2),
        }

        # Changes input ret_hist like in Root
        self.hist = new_hist
        self._calc_stats()

    def ret_asnumpyhist(self):
        """Return histogram in np.histogram format from current histogram.

        Note: Cannot use data, as this doesn't include added histograms etc."""

        return self.hist["y"].values, self.hist["x_edges"], self.hist["y_edges"]


def main():
    """Example for Bi"""
    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(dir)
    dataloader.auto()

    filt_meas = dataloader.ret_filt_meas(
        ["tp", "src", "nomad_no", "cyc_no"], [1, 2, 70372, 194348]
    )[0]

    dt_facs = []
    cyc_facs = []
    for i in [0, 1]:
        datacop = RootPerkeo(filt_meas.file_list[i])
        datacop.set_filtdef()
        datacop.auto()

        dt_facs.append(datacop.dt_fac)
        cyc_facs.append(datacop.cy_valid_no)

    corr_class = CorrPerkeo(filt_meas, mode=1)
    corr_class.set_all_corr(bactive=False)
    corr_class.corrections["Drift"] = True
    corr_class.corrections["Scan2D"] = True
    corr_class.corrections["RateDepElec"] = True
    corr_class.corrections["Pedestal"] = True
    corr_class.corrections["DeadTime"] = True

    for det in [0, 1]:
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 0,
                "up_lim": 4e9,
                "index": det,
            }
        )
    corr_class.corr(bstore=True, bwrite=False, bconcat=False)
    histp = corr_class.histograms[0]
    # Calling input data for HistPerkeo created in CorrPerkeo includes all
    # corrections on single events, i.e., not DeadTime and Bg subtraction.
    det_0 = histp[1][0].ret_data()
    det_1 = histp[1][1].ret_data()

    data2d = np.concatenate(
        [det_0.reshape(1, det_0.shape[0]), det_1.reshape(1, det_1.shape[0])]
    )
    hist_p0 = Hist2DPerkeo(
        data2d,
        bin_count=[75, 75],
    )
    hist_p0.plot_hist(rng=[0.0, 45e3, 0.0, 45e3])

    # Redoing for background
    filt_meas.tp = 2
    filt_meas.file_list = [filt_meas.file_list[1]]
    filt_meas.date_list = [filt_meas.date_list[1]]

    corr_class = CorrPerkeo(filt_meas, mode=1)
    corr_class.set_all_corr(bactive=False)
    corr_class.corrections["Drift"] = True
    corr_class.corrections["Scan2D"] = True
    corr_class.corrections["RateDepElec"] = True
    corr_class.corrections["Pedestal"] = True
    corr_class.corrections["DeadTime"] = True

    for det in [0, 1]:
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 0,
                "up_lim": 4e9,
                "index": det,
            }
        )
    corr_class.corr(bstore=True, bwrite=False, bconcat=False)
    histp = corr_class.histograms[0]
    det_0 = histp[1][0].ret_data()
    det_1 = histp[1][1].ret_data()

    data2d = np.concatenate(
        [det_0.reshape(1, det_0.shape[0]), det_1.reshape(1, det_1.shape[0])]
    )
    hist_p1 = Hist2DPerkeo(
        data2d,
        bin_count=[75, 75],
    )
    hist_p1.plot_hist(rng=[0.0, 45e3, 0.0, 45e3])

    # Calculating corrected 2D histogram
    hist_p0.scal(dt_facs[0])
    hist_p1.scal(dt_facs[1] * cyc_facs[0] / cyc_facs[1])
    hist_p0.addhist(hist_p1, -1.0)
    hist_p0.plot_hist(rng=[0.0, 45e3, 0.0, 45e3])


if __name__ == "__main__":
    main()
