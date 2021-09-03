"""Unit test for basic histogram creation, subtraction and fitting with errors."""

import numpy as np

import panter.config.evalFitSettings as eFS
from panter.data.dataHistPerkeo import HistPerkeo
from panter.eval.evalFit import DoFit
from tests.unittestroot import UnitTestRoot


class HistTestFit(UnitTestRoot):
    """Unit test class for basic histogram creation, subtraction and fitting.

    Inherited from base class UnitTestRoot.

    Parameters
    ----------
    txtfile: str
        Name of sample txt file with data to fill the histogram with.
    params: [int, float, float, float, float]
        List of parameters to be used for histograms and fit:
        [BinCounts, HistLowLim, HistUpLim, FitRangeLow, FitRangeUp]
    """

    def __init__(self, txtfile: str, params: list):
        self._txtfile = txtfile
        self.hist_par = params[0:3]
        self.fit_par = params[3:]
        self._root_macro = "basicfit.cpp"
        super().__init__(
            test_label="HistTestFit", params=params, root_macro=self._root_macro
        )

    def _do_root(self):
        """Do ROOT evaluation part for the unit test."""
        return super()._do_root([self._txtfile], self._params)

    def _do_panter(self):
        """Do panter evaluation part for the unit test."""

        data_raw = open(self._txtfile).read().split()
        data_raw = list(map(float, data_raw))

        hpanter1 = HistPerkeo(*[data_raw, *self.hist_par])
        hpanter2 = HistPerkeo(*[np.array(data_raw) + 2, *self.hist_par])
        hpanter1.addhist(hpanter2, -0.5)

        fitclass = DoFit(hpanter1.hist)
        fitclass.setup(eFS.pol0)
        fitclass.limit_range(self.fit_par)

        fitres = fitclass.fit()

        panter_fitres = [
            fitres.params["c0"].value,
            fitres.params["c0"].stderr,
            fitclass.ret_gof()[0],
        ]

        return np.asarray(panter_fitres)


def do_histtestfit() -> bool:
    """Run this unit test with hard coded, default parameters."""

    file = "sample.txt"
    par = [5, 0, 15, 0, 15]

    test = HistTestFit(txtfile=file, params=par)
    return test.test(brel_dev=False, bprint=True)
