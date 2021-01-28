"""Unit test for histogram writing to root files with errors. Based on histogram.py"""

import numpy as np
import uproot
import subprocess
import panter.core.dataPerkeo as dP
from tests.unittestroot import UnitTestRoot


class HistTestOut(UnitTestRoot):
    """Unit test class for histogram writing test with errors.

    Inherited from base class UnitTestRoot and same as histogram.py with output.
    Uses same root macro as histogram.py

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
            test_label="HistTestOut", params=params, root_macro=self._root_macro
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
        # export
        hpanter1.write2root("hpanter", "inter")
        # re-simport for comparisson
        himport = uproot.open("inter.root")
        hpanter_imp = himport["hpanter"]
        bincent = []
        for j in range(hpanter_imp.edges.size - 1):
            bincent.append(0.5 * (hpanter_imp.edges[j] + hpanter_imp.edges[j + 1]))
        result = []
        for i in range(len(bincent)):
            result.append(bincent[i])
            result.append(hpanter_imp.values[i])
            result.append(np.sqrt(hpanter_imp.variances)[i])

        subprocess.run(["rm", "inter.root"])

        return np.asarray(result)


def do_histtestout() -> bool:
    """Run this unit test with hard coded, default parameters."""

    file = "sample.txt"
    par = [5, 0, 15]

    test = HistTestOut(txtfile=file, params=par)
    return test.test(False)
