""""""

import numpy as np
import panter.core.dataPerkeo as dP

from tests.unittestroot import UnitTestRoot


class HistTestBasic(UnitTestRoot):
    """"""

    def __init__(self, txtfile: str, params: list, root_macro: str):
        super().__init__(
            test_label="HistTestBasic", params=params, root_macro=root_macro
        )
        self.txtfile = txtfile

    def do_panter(self):
        data_raw = open(self.txtfile).read().split()
        data_raw = list(map(float, data_raw))

        hpanter1 = dP.HistPerkeo(*[data_raw, *self.params])
        hpanter2 = dP.HistPerkeo(*[np.array(data_raw) + 2, *self.params])
        hpanter1.addhist(hpanter2, -0.5)

        return hpanter1.hist.to_numpy().flatten()

    def do_root(self):
        return super().do_root([self.txtfile], self.params)


def do_histtestbasic():
    """"""

    file = "sample.txt"
    par = [5, 0, 15]
    root_mac = "histogram.cpp"

    test = HistTestBasic(txtfile=file, params=par, root_macro=root_mac)
    return test.test(False)
