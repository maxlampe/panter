""""""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.pyplot import figure
from panter.base.histBase import HistBase
from panter.data.dataMisc import ret_hist

output_path = os.getcwd()
plt.rcParams.update({"font.size": 12})


class HistPerkeo(HistBase):
    """Histogram object for use with PERKEO data.

    Takes data and histogram parameters to create a histogram with
    basic helper functions like plotting and histogram arithmetics.
    Does over- and underflow bins "correctly", unlike numpy.hist.
    Designed to be similar to ROOT histogram class and build upon
    numpy.hist.

    Parameters
    ----------
    data : np.array
        Data to be histogrammed.
    bin_count, low_lim, up_lim: int
        Histogram parameters: Bin count, upper and lower limit

    Attributes
    ----------
    n_events, mean, stdv, stats, parameters, bin_width, integral
        Set of histogram properties and information.
    parameters: dict
        Histogram parameters as dict.
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
        super().__init__(data=data)

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
        self.bin_width = (up_lim - low_lim) / bin_count

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
        bxlog: bool = False,
        bylog: bool = False,
        filename: str = "",
        fsize: int = None,
    ):
        """Plot histogram.

        Parameters
        ----------
        rng: list
            Plot ranges as list [x0, x1, y0, y1].
        title, xlabel, ylabel: str, str, str
            Plot title and axis labels.
        bsavefig, bxlog, bylog: False, False, False
            Bools to set whether to write the plot to a file or to set axis to log
            scales.
        fsize: int
            Font size for plot.
        filename: str
            Output file name as {filename}.pdf. Requires bsavefig = True.
        """

        figure(figsize=(8, 6))
        plt.errorbar(self.hist["x"], self.hist["y"], self.hist["err"], fmt=".")
        if rng is not None:
            plt.axis([rng[0], rng[1], rng[2], rng[3]])
        if self.stats["std"] is None:
            self.stats["std"] = 0.0

        plt.title(title, fontsize=fsize)
        plt.ylabel(ylabel, fontsize=fsize)
        plt.xlabel(xlabel, fontsize=fsize)
        plt.tick_params(labelsize=fsize)
        plt.annotate(
            (
                f"n_ev = {self.stats['noevents']:0.1f}\n"
                + f"Mean = {self.stats['mean']:0.2f}\n"
                + f"StDv = {self.stats['std']:0.2f}"
            ),
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="1"),
        )
        if bylog:
            plt.yscale("log")
        if bxlog:
            plt.xscale("log")

        if bsavefig:
            if filename == "":
                filename = "histperkeo"
            plt.savefig(f"{output_path}/{filename}.pdf", dpi=300)
        plt.show()

    def addhist(self, hist_p: HistPerkeo, fac: float = 1.0):
        """Add another histogram to existing one with multiplicand.

        Parameters
        ----------
        hist_p: HistPerkeo
            Histogram to multiply y values with (bin by bin), including errors.
        fac: float
            Scaling factor to multiply bin count (y-value and error) of (external)
            histogram hist_p.
        """

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
        """Divide by another histogram.

        Parameters
        ----------
        hist_p: HistPerkeo
            Histogram to divide y values with (bin by bin), including errors.
        """

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
        """Multiply by another histogram.

        Parameters
        ----------
        hist_p: HistPerkeo
            Histogram to multiply y values with (bin by bin), including errors.
        """

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
        """Scale histogram by a factor.

        Parameters
        ----------
        fac: float
            Scaling factor to multiply bin count (y-value and error) with.
        """

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

    def ret_asnumpyhist(self):
        """Return histogram in np.histogram format from current histogram.

        Note: Cannot use data, as this doesnt include added histograms etc."""

        deltx = 0.5 * (self.hist["x"].values[1] - self.hist["x"].values[0])
        binedge = self.hist["x"].values - deltx
        binedge = np.append(binedge, self.hist["x"].values[-1] + deltx)

        return self.hist["y"].values, binedge

    def ret_data(self):
        """Return original input data"""
        return self._data

    def write2root(
        self, histname: str, filename: str, out_dir: str = None, bupdate: bool = False
    ):
        """Write the histogram into a root file.

        Parameters
        ----------
        histname: str
            Name of histogram (first two parameters of TH1F in ROOT)
        filename: str
            Name of file as {filename}.root
        out_dir: str
            Path to output directory as {out_dir}/
        bupdate: False
            Flag to "UPDATE" or "RECREATE" file in TFile class.
        """

        assert self._bfound_root, "ERROR: Could not find ROOT package."
        from ROOT import TFile, TH1F

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
