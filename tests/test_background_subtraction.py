"""Unit test for background subtraction in beta spec + fitting a constant to high E."""

import configparser

import numpy as np

from panter.config import conf_path
from panter.config.evalFitSettings import pol0
from panter.data.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.eval.corrPerkeo import CorrPerkeo
from panter.eval.evalFit import DoFit
from tests.unittestroot import UnitTestRoot


class BackgroundFitTest(UnitTestRoot):
    """Unit test class for background subtraction and constant fit.

    Inherits from UnitTestRoot. Takes root file of PERKEO beam measurement, creates
    signal spectra via background subtraction and fits a constant to very high energies.
    Runs over several files and only if unit test is passed for all files, it will
    return a final pass.

    Parameters
    ----------
    dirname : str
        Name of data directory.
    params: [int, float, float, float, float]
        List of parameters to be used for histograms and fit:
        [BinCounts, HistLowLim, HistUpLim, FitRangeLow, FitRangeUp]

    Attributes
    ----------
    dataloader : DLPerkeo
        Data loader for data directory.
    files_2do : [900, 903]
        Iterator for defining batches from data loader.
    batch : np.array(MeasPerkeo)
        Array of filtered MeasPerkeo from data loader.
    """

    def __init__(self, dirname: str, params: list):
        self._dirname = dirname
        self._root_macro = "background_subtraction.cpp"
        super().__init__(
            test_label="BackgroundFitTest", params=params, root_macro=self._root_macro
        )

        self.dataloader = DLPerkeo(self._dirname)
        self.dataloader.auto()
        self.files_2do = [900, 903]
        self.batch = self.dataloader.ret_filt_meas(["src"], [5])[
            self.files_2do[0] : self.files_2do[1]
        ]

    def _do_root(self, filename: str):
        """Do ROOT evaluation part for the unit test."""
        return super()._do_root([filename], self._params)

    def _do_panter(self, meas: MeasPerkeo):
        """Do panter evaluation part for the unit test."""

        fit_range = [self._params[3], self._params[4]]
        corr_class = CorrPerkeo(dataloader=meas, mode=1, bonlynew=True)
        corr_class.set_all_corr(bactive=False)
        corr_class.corr(bstore=True, bwrite=False)

        panter_fitres = []
        for hist in corr_class.histograms[0, 1]:
            fitclass = DoFit(hist.hist)
            # hist.plot_hist([48000, 51500, -40, 40])
            fitclass.setup(pol0)
            fitclass.limit_range([fit_range[0], fit_range[1]])
            fitres = fitclass.fit()

            panter_fitres.append(fitres.params["c0"].value)
            panter_fitres.append(fitres.params["c0"].stderr)
            panter_fitres.append(fitclass.ret_gof()[0])

        # hot-fix to exclude total sum spectra
        panter_res = np.asarray(panter_fitres)[0:6]

        return panter_res

    def test(self) -> bool:
        """Run this unit test."""

        results = []
        for ind, meas in enumerate(self.batch):
            file = meas()[2][0]
            root_res = self._do_root(file)
            panter_res = self._do_panter(np.asarray([meas]))
            # print(panter_res)
            results.append(super()._check(root_res, panter_res))

        no_passed = np.asarray(results).sum()
        if no_passed == len(self.batch):
            print(f"GREAT SUCCESS: Unit test passed for all {len(self.batch)} files.")
        else:
            print(f"FAILURE: Only {no_passed} of {len(self.batch)} passed")

        return no_passed == len(self.batch)


def test_backgroundfittest() -> bool:
    """Run this unit test with hard coded, default parameters."""

    cnf = configparser.ConfigParser()
    cnf.read(f"{conf_path}/evalRaw.ini")
    histsum_par = [
        int(cnf["dataPerkeo"]["SUM_hist_counts"]),
        int(cnf["dataPerkeo"]["SUM_hist_min"]),
        int(cnf["dataPerkeo"]["SUM_hist_max"]),
    ]

    dirname = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    par = [histsum_par[0], histsum_par[1], histsum_par[2], 48000, 51500]

    test = BackgroundFitTest(dirname=dirname, params=par)
    res = test.test()
    assert res

    return res


# do_backgroundfittest()
