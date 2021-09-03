"""Unit test for histogram creation using PERKEO data and applying filters."""

import panter.data.dataMisc as dP
from tests.unittestroot import UnitTestRoot


class HistTestFilter(UnitTestRoot):
    """Unit test class for histogram creation using PERKEO data and applying filters.

    Inherited from base class UnitTestRoot.

    Parameters
    ----------
    rootfile: str
        Name of PERKEO ROOT file with data to fill the histogram with.
    params: [int, float, float, bool]
        List of parameters to be used for histograms:
        [BinCounts, HistLowLim, HistUpLim, ApplyFilter]
    """

    def __init__(self, rootfile: str, params: list):
        self._rootfile = rootfile
        self._bfilter = params[3]
        self._root_macro = "filter.cpp"
        super().__init__(
            test_label="HistTestFilter", params=params, root_macro=self._root_macro
        )

    def _do_root(self):
        """Do ROOT evaluation part for the unit test."""
        return super()._do_root([self._rootfile], self._params)

    def _do_panter(self):
        """Do panter evaluation part for the unit test."""

        data = dP.RootPerkeo(self._rootfile)
        data.info()
        if self._bfilter:
            data.set_filtdef()
            data.set_filt(
                "data",
                fkey="DeltaTriggerTime",
                active=True,
                ftype="num",
                low_lim=380e3,
                up_lim=650e3,
            )
            data.auto(1)
        else:
            data.auto()

        hist_par = {
            "bin_count": self._params[0],
            "low_lim": self._params[1],
            "up_lim": self._params[2],
        }
        data.gen_hist(lpmt=range(data.no_pmts), cust_hist_par=hist_par)
        hpanter1 = data.hists[0]

        return hpanter1.hist.to_numpy().flatten()


def do_histtestfilter() -> bool:
    """Run this unit test with hard coded, default parameters."""

    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    file = dir + "data185836-70086_beam.root"
    par1 = [3, 0, 500, 0]
    par2 = [3, 0, 500, 1]

    test_wo_filter = HistTestFilter(rootfile=file, params=par1)
    restult_wo_filter = test_wo_filter.test(brel_dev=False, bprint=True)

    test_wi_filter = HistTestFilter(rootfile=file, params=par2)
    restult_wi_filter = test_wi_filter.test(brel_dev=False, bprint=True)

    passed = restult_wo_filter and restult_wi_filter
    if passed:
        print(f"GREAT SUCCESS: Unit test passed with and without filter. ")
    else:
        print(
            f"FAILURE: Unit test not passed. Result with and without filter: "
            + f"{restult_wo_filter} / {restult_wi_filter}"
        )

    return passed
