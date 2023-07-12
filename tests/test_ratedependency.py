""" Unit tests for rate dependency correction """

import pandas as pd
import numpy as np
import unittest

try:
    from panter.data.dataMeasPerkeo import MeasPerkeo
    from panter.eval.corrPerkeo import CorrPerkeo
    from panter.eval.evalFit import DoFit
    from panter.config.evalFitSettings import gaus_simp
    from panter.config.params import delt_pmt
except ImportError:
    print("Panter was not succesfully installed")
    exit()


class TestPedestal(unittest.TestCase):
    """Unit test class to fit the summed peak from all PMTs
     for two oscillating peaks with custom generated datafile.
     Only works for simple signals.


    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super(TestPedestal, self).__init__(*args, **kwargs)
        self._rootfile = "test_ratedependency.root"
        self._measurement = MeasPerkeo(2, 5, list([self._rootfile]))
        self._signal_peaks = [100, 3000]
        self._signal_widths = [50, 50]
        self._num_PMTs = 16
        self._deltatime = 100
        self._k = 99.0
        self._expfrac = np.exp(-self._deltatime / self._k)
        self._delta = 0.0033

    def test_singlePMT_correction(self):
        """test each ratedependency corrected PMT signals"""
        corr_class = CorrPerkeo(self._measurement, mode=2)
        corr_class.set_all_corr(bactive=False)
        corr_class.corrections["RateDepElec"] = True
        corr_class.corr(bstore=True, bwrite=False)
        for pmt in range(self._num_PMTs):
            testhist = corr_class.histograms[0][1][pmt]
            # testhist.plot_hist()
            for signal in range(2):
                if signal == 0:
                    mu = (
                        self._signal_peaks[0]
                        + self._signal_peaks[1] * delt_pmt[pmt] * self._expfrac
                    )
                elif signal == 1:
                    mu = (
                        self._signal_peaks[1]
                        + self._signal_peaks[0] * delt_pmt[pmt] * self._expfrac
                    )
                width = self._signal_widths[signal]
                fitclass = DoFit(testhist.hist)
                fitclass.setup(gaus_simp)
                fitclass.limit_range([mu - 3 * width, mu + 3 * width])
                fitclass.set_bool("boutput", False)
                fitclass.set_fitparam("mu", mu)
                fitclass.set_fitparam("sig", width)
                # fitclass.set_fitparam("norm", 4000.0)
                fitclass.fit()
                fitted_mu = fitclass.ret_results().params["mu"].value
                fitted_mu_err = fitclass.ret_results().params["mu"].stderr
                lower_intervall = mu - 3 * fitted_mu_err  # TODO precision!
                upper_intervall = mu + 3 * fitted_mu_err  # TODO precision!
                self.assertGreaterEqual(fitted_mu, lower_intervall)
                self.assertLessEqual(fitted_mu, upper_intervall)
