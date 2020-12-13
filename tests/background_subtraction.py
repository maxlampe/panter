""""""

from tests.unittestroot import UnitTestRoot
import configparser
from panter.config import conf_path

import numpy as np
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS


class BackgroundFitTest(UnitTestRoot):
    """"""

    def __init__(self, dirname: str, params: list, root_macro: str):
        super().__init__(
            test_label="BackgroundFitTest", params=params, root_macro=root_macro
        )
        self.dirname = dirname
        self.files_2do = 3
        self.dataloader = DLPerkeo(self.dirname)
        self.dataloader.auto()
        self.batches = self.dataloader.ret_filt_meas(["src"], [5])[
            900 : 900 + self.files_2do
        ]

    def do_root(self, filename: str):
        return super().do_root([filename], self.params)

    def do_panter(self, meas: MeasPerkeo):
        """"""

        fit_range = [self.params[3], self.params[4]]
        corr_class = corrPerkeo(meas)
        corr_class.corrections["Pedestal"] = False
        corr_class.corrections["RateDepElec"] = False
        corr_class.corr(bstore=True, bwrite=False)

        panter_fitres = []
        for hist in corr_class.histograms[0, 0]:
            fitclass = eP.DoFit(hist.hist)
            fitclass.setup(eFS.pol0)
            fitclass.limitrange([fit_range[0], fit_range[1]])
            fitres = fitclass.fit()

            panter_fitres.append(fitres.params["c0"].value)
            panter_fitres.append(fitres.params["c0"].stderr)
            panter_fitres.append(fitclass.ret_gof()["rChi2"])

        return np.asarray(panter_fitres)

    def test(self):
        """"""

        results = []
        for ind, meas in enumerate(self.batches):
            file = meas()[2][0]
            root_res = self.do_root(file)
            panter_res = self.do_panter(meas)

            results.append(super().check(root_res, panter_res))

        no_passed = np.asarray(results).sum()
        if no_passed == len(self.batches):
            print(f"GREAT SUCCESS: Unit test passed for all {len(self.batches)} files.")
        else:
            print(f"FAILURE: Only {no_passed} of {len(self.batches)} passed")

        return no_passed == len(self.batches)


def do_backgroundfittest():
    """"""

    cnf = configparser.ConfigParser()
    cnf.read(f"{conf_path}/evalRaw.ini")
    histsum_par = [
        int(cnf["dataPerkeo"]["SUM_hist_counts"]),
        int(cnf["dataPerkeo"]["SUM_hist_min"]),
        int(cnf["dataPerkeo"]["SUM_hist_max"]),
    ]

    dirname = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    par = [histsum_par[0], histsum_par[1], histsum_par[2], 45000, 51500]
    root_mac = "background_subtraction.cpp"

    test = BackgroundFitTest(dirname=dirname, params=par, root_macro=root_mac)

    return test.test()
