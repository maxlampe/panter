""" Unit tests for summed signal (detsum) """

import pandas as pd
import numpy as np
import unittest

try:
    from panter.data.dataMeasPerkeo import MeasPerkeo
    from panter.eval.corrPerkeo import CorrPerkeo
    from panter.eval.evalFit import DoFit
    from panter.config.evalFitSettings import gaus_simp
except ImportError:
    print("Panter was not succesfully installed")
    exit()


class TestPedestal(unittest.TestCase):
    """Unit test class to fit the summed data from all PMTs
     with custom generated datafile. Only works for
     simple signal.


    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super(TestPedestal, self).__init__(*args, **kwargs)
        self._rootfile = "test_detsum.root"
        self._measurement = MeasPerkeo(2, 5, list([self._rootfile]))
        self._signal_peak = 1000
        self._signal_width = 50
        self._num_PMTs = 16

    def test_sum_correction(self):
        """test summend signal for all PMT"""
        custom_bins = {"bin_count": 1000, "low_lim": 0, "up_lim": 20000}
        corr_class = CorrPerkeo(
            self._measurement, mode=0, custom_sum_hist_par=custom_bins
        )
        corr_class.set_all_corr(bactive=False)
        corr_class.corr(bstore=True, bwrite=False)
        testhist = corr_class.histograms[0][1][0]
        # testhist.plot_hist()

        mu = self._signal_peak * self._num_PMTs
        width = np.sqrt((self._signal_width**2) * self._num_PMTs)
        fitclass = DoFit(testhist.hist)
        fitclass.setup(gaus_simp)
        fitclass.limit_range([mu - 2 * width, mu + 2 * width])
        fitclass.set_bool("boutput", False)
        fitclass.set_fitparam("mu", mu)
        fitclass.set_fitparam("sig", width)
        fitclass.fit()

        fitted_mu = fitclass.ret_results().params["mu"].value
        fitted_mu_err = fitclass.ret_results().params["mu"].stderr
        lower_intervall = mu - 1.5 * fitted_mu_err
        upper_intervall = mu + 1.5 * fitted_mu_err
        self.assertGreaterEqual(fitted_mu, lower_intervall)
        self.assertLessEqual(fitted_mu, upper_intervall)

        fitted_sig = fitclass.ret_results().params["sig"].value
        fitted_sig_err = fitclass.ret_results().params["sig"].stderr
        lower_intervall_w = width - 1.5 * fitted_sig_err
        upper_intervall_w = width + 1.5 * fitted_sig_err
        self.assertGreaterEqual(abs(fitted_sig), lower_intervall_w)
        self.assertLessEqual(abs(fitted_sig), upper_intervall_w)
