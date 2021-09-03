"""Unit test for basic histogram creation and subtraction with errors"""

import numpy as np
import panter.data.dataPerkeo as dP
from tests.unittestroot import UnitTestRoot


class HistTestBasic(UnitTestRoot):
    """Unit test class for basic histogram creation and subtraction with errors.

    Inherited from base class UnitTestRoot.

    Parameters
    ----------
    txtfile: str
        Name of sample txt file with data to fill the histogram with.
    params: [int, float, float]
        List of parameters to be used for histograms: [BinC, LowLim, UpLim]
    """

    def __init__(self, txtfile: str, params: list):
        self._root_macro = "histogram.cpp"
        self._txtfile = txtfile
        super().__init__(
            test_label="HistTestBasic", params=params, root_macro=self._root_macro
        )

    def _do_root(self):
        """Do ROOT evaluation part for the unit test."""
        return super()._do_root([self._txtfile], self._params)

    def _do_panter(self):
        """Do panter evaluation part for the unit test."""

        data_raw = open(self._txtfile).read().split()
        data_raw = list(map(float, data_raw))

        hpanter1 = dP.HistPerkeo(*[data_raw, *self._params])
        hpanter2 = dP.HistPerkeo(*[np.array(data_raw) + 2, *self._params])
        hpanter1.addhist(hpanter2, -0.5)

        return hpanter1.hist.to_numpy().flatten()


def do_histtestbasic() -> bool:
    """Run this unit test with hard coded, default parameters."""

    file = "sample.txt"
    par = [5, 0, 15]

    test = HistTestBasic(txtfile=file, params=par)
    return test.test(False)
