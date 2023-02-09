""""""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure

from panter.data.dataMisc import ret_hist

output_path = os.getcwd()
plt.rcParams.update({"font.size": 12})

bfound_root = True
try:
    from ROOT import TFile, TH1F
except ModuleNotFoundError:
    bfound_root = False


class HistPerkeo:
    """Histogram object for use with PERKEO data.

    Takes data and histogram parameters to create a histogram with
    basic helper functions.

    Parameters
    ----------
    data : np.array
    bin_count, low_lim, up_lim: int
        Histogram parameters: Bin count, upper and lower limit

    Attributes
    ----------
    stats
    parameters
        see above section
    hist : pd.DataFrame
        Returned histogram from function ret_hist()

    Examples
    --------
    Create a histogram with any np.array of data and plot the result:

    >>> histogram = HistPerkeo(data=data_array, bin_count=10, low_lim=-10, up_lim=10)
    >>> histogram.plot_hist()
    """

    def __init__(
        self,
        data: np.array(float),
        bin_count: int = 1024,
        low_lim: int = 0,
        up_lim: int = 52000,
    ):
        assert low_lim <= up_lim, "Error: lower limit bigger than upper limit."

        self._data = np.asarray(data)
        if self._data.shape != ():
            self.n_events = self._data.shape[0]
            self.mean = self._data.mean()
            self.stdv = self._data.std()
            self.stats = {
                "mean": self.mean,
                "std": self.stdv,
                "noevents": self.n_events,
            }
        self.parameters = {"bin_count": bin_count, "low_lim": low_lim, "up_lim": up_lim}
        self.bin_count = bin_count
        self.up_lim = up_lim
        self.low_lim = low_lim
        self.bin_width = (self.up_lim - self.low_lim) / self.bin_count

        if self._data.shape != ():
            self.hist = ret_hist(self._data, **self.parameters)
        else:
            self.hist = None

        self.integral = (self.hist["y"] * self.bin_width).sum()

    def _calc_stats(self):
        """Calculate mean and biased variance of histogram based on bin content."""

        self.n_events = self.hist["y"].sum()
        self.mean = (self.hist["x"] * self.hist["y"]).sum() / self.n_events
        var = ((self.hist["x"] - self.mean) ** 2 * self.hist["y"]).sum() / self.n_events
        self.stdv = np.sqrt(var)

        self.stats = {
            "mean": self.mean,
            "std": self.stdv,
            "noevents": self.n_events,
        }

        self.integral = (self.hist["y"] * self.bin_width).sum()

    def plot_hist(
        self,
        rng: list = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        bsavefig: bool = False,
        filename: str = "",
    ):
        """Plot histogram."""

        figure(figsize=(8, 6))
        plt.errorbar(self.hist["x"], self.hist["y"], self.hist["err"], fmt=".")
        if rng is not None:
            plt.axis([rng[0], rng[1], rng[2], rng[3]])
        if self.stats["std"] is None:
            self.stats["std"] = 0.0

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.annotate(
            f"Mean = {self.stats['mean']:0.2f}\n" f"StDv = {self.stats['std']:0.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="1"),
        )

        if bsavefig:
            if filename == "":
                filename = "histperkeo"
            plt.savefig(f"{output_path}/{filename}.png", dpi=300)
        plt.show()

    def addhist(self, hist_p: HistPerkeo, fac: float = 1.0):
        """Add another histogram to existing one with multiplicand."""

        assert self.parameters == hist_p.parameters, "ERROR: Binning does not match."

        newhist = pd.DataFrame(
            {
                "x": self.hist["x"],
                "y": (self.hist["y"] + fac * hist_p.hist["y"]),
                "err": np.sqrt(self.hist["err"] ** 2 + (fac * hist_p.hist["err"]) ** 2),
            }
        )
        # Changes input ret_hist like in Root
        self.hist = newhist
        self._calc_stats()

    def divbyhist(self, hist_p: HistPerkeo):
        """Divide by another histogram."""

        assert self.parameters == hist_p.parameters, "ERROR: Binning does not match."

        filt = hist_p.hist["y"] != 0.0
        hist_p.hist = hist_p.hist[filt]
        self.hist = self.hist[filt]

        newhist = pd.DataFrame(
            {
                "x": self.hist["x"],
                "y": (self.hist["y"] / hist_p.hist["y"]),
                "err": np.sqrt(
                    (self.hist["err"] / hist_p.hist["y"]) ** 2
                    + (self.hist["err"] * hist_p.hist["err"] / (hist_p.hist["y"] ** 2))
                    ** 2
                ),
            }
        )

        # Changes input ret_hist like in Root
        self.hist = newhist
        self._calc_stats()

    def multbyhist(self, hist_p: HistPerkeo):
        """Multiply by another histogram."""

        assert self.parameters == hist_p.parameters, "ERROR: Binning does not match."

        # filt = hist_p.hist["y"] != 0.0
        # hist_p.hist = hist_p.hist[filt]
        # self.hist = self.hist[filt]

        newhist = pd.DataFrame(
            {
                "x": self.hist["x"],
                "y": (self.hist["y"] * hist_p.hist["y"]),
                "err": np.sqrt(
                    (self.hist["err"] * hist_p.hist["y"]) ** 2
                    + (self.hist["y"] * hist_p.hist["err"]) ** 2
                ),
            }
        )

        # Changes input ret_hist like in Root
        self.hist = newhist
        self._calc_stats()

    def scal(self, fac: float):
        """Scale histogram by a factor."""

        newhist = pd.DataFrame(
            {
                "x": self.hist["x"],
                "y": (self.hist["y"] * fac),
                "err": np.sqrt((fac * self.hist["err"]) ** 2),
            }
        )
        # Changes input ret_hist like in Root
        self.hist = newhist
        self._calc_stats()

    def norm_hist(self):
        """Norm histogram to sum up to 1"""

        self._calc_stats()
        self.scal(1./self.integral)

    def ret_asnumpyhist(self):
        """Return histogram in np.histogram format from current histogram.

        Note: Cannot use data, as this doesnt include added histograms etc."""

        deltx = 0.5 * (self.hist["x"].values[1] - self.hist["x"].values[0])
        binedge = self.hist["x"].values - deltx
        binedge = np.append(binedge, self.hist["x"].values[-1] + deltx)

        return self.hist["y"].values, binedge

    def write2root(
        self, histname: str, filename: str, out_dir: str = None, bupdate: bool = False
    ):
        """Write the histogram into a root file."""

        assert bfound_root, "ERROR: Could not find ROOT package."

        if out_dir is None:
            out_dir = output_path

        opt = "UPDATE" if bupdate else "RECREATE"
        hfile = TFile(f"{out_dir}/{filename}.root", opt, "Panter Output")
        rhist = TH1F(
            f"{histname}",
            f"{histname}",
            self.parameters["bin_count"],
            self.parameters["low_lim"],
            self.parameters["up_lim"],
        )
        for i in range(1, rhist.GetNbinsX()):
            rhist.SetBinContent(i, self.hist["y"][i - 1])
            rhist.SetBinError(i, self.hist["err"][i - 1])
        rhist.Draw()
        hfile.Write()

        return 0
