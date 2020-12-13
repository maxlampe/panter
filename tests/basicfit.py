""""""


import numpy as np
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from tests.unittestroot import UnitTestRoot


class HistTestFit(UnitTestRoot):
    """"""

    def __init__(self, txtfile: str, params: dict, root_macro: str):
        super().__init__(test_label="HistTestFit", params=params, root_macro=root_macro)
        self.txtfile = txtfile
        self.hist_par = params[0:3]
        self.fit_par = params[3:]

    def do_panter(self):
        data_raw = open(self.txtfile).read().split()
        data_raw = list(map(float, data_raw))

        hpanter1 = dP.HistPerkeo(*[data_raw, *self.hist_par])
        hpanter2 = dP.HistPerkeo(*[np.array(data_raw) + 2, *self.hist_par])
        hpanter1.addhist(hpanter2, -0.5)

        # do fit on hpanter1
        fitclass = eP.DoFit(hpanter1.hist)
        fitclass.setup(eFS.pol0)
        fitclass.limitrange(self.fit_par)

        fitres = fitclass.fit()

        panter_fitres = [
            fitres.params["c0"].value,
            fitres.params["c0"].stderr,
            fitclass.ret_gof()["rChi2"],
        ]

        return np.asarray(panter_fitres)

    def do_root(self):
        return super().do_root([self.txtfile], self.params)


def do_histtestfit():
    """"""

    file = "sample.txt"
    par = [5, 0, 15, 0, 15]
    root_mac = "basicfit.cpp"

    test = HistTestFit(txtfile=file, params=par, root_macro=root_mac)
    return test.test(brel_dev=False, bprint=True)
