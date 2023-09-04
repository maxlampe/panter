"""Base class for histograms 1D/2D"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

output_path = os.getcwd()
plt.rcParams.update({"font.size": 12})


class HistBase:
    """Histogram base class for use with PERKEO data.

    Parameters
    ----------
    data: np.array
        Data to be histogrammed.

    Attributes
    ----------
    integral
        Integral of histogram.
    """

    def __init__(
        self,
        data: np.array,
    ):
        self._data = np.asarray(data)
        self.integral = None
        self._bfound_root = True
        try:
            import ROOT
        except ModuleNotFoundError:
            self._bfound_root = False

    def _calc_stats(self):
        """Calculate mean and biased variance of histogram based on bin content."""
        pass

    def plot_hist(self):
        """Plot histogram."""
        pass

    def addhist(self):
        """Add another histogram to existing one with multiplicand."""
        pass

    def divbyhist(self):
        """Divide by another histogram."""
        pass

    def multbyhist(self):
        """Multiply by another histogram."""
        pass

    def scal(self, fac: float):
        """Scale histogram by a factor.

        Parameters
        ----------
        fac: float
            Scaling factor to multiply bin count (y-value) with.
        """
        pass

    def norm_hist(self):
        """Norm histogram to sum up to 1"""

        # Re-calculate integral to be sure.
        self._calc_stats()
        self.scal(1.0 / self.integral)

    def ret_asnumpyhist(self):
        """Return histogram in np.histogram format from current histogram.

        Note: Cannot use data, as this doesnt include added histograms etc."""
        pass

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
        pass
