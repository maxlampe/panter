"""Unit tests for pedestal determination and correction"""

import unittest
import pandas as pd
import numpy as np


try:
    from panter.data.dataRootPerkeo import RootPerkeo
    from panter.data.dataMeasPerkeo import MeasPerkeo
    from panter.eval.corrPerkeo import CorrPerkeo
    from panter.eval.pedPerkeo import PedPerkeo
    from panter.eval.evalFit import DoFit
    from panter.config.evalFitSettings import gaus_simp
except ImportError:
    print("Panter was not succesfully installed")
    exit()


class TestPedestal(unittest.TestCase):
    """Unit test class to fit and correct data from pedestals
     with custom generated datafile. Only works for
     simple signal and pedestal data.


    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super(TestPedestal, self).__init__(*args, **kwargs)
        self._rootfile = "pedestal.root"
        self._data = RootPerkeo(self._rootfile)
        self._custom_bins = {"bin_count": 1000, "low_lim": -500, "up_lim": 500}
        self._pedtest = PedPerkeo(
            self._data, bplot_fit=False, custom_hist_par=self._custom_bins
        )
        self._pedestals = self._pedtest.ret_pedestals()
        self._pedestals_dim = np.shape(self._pedestals)
        self._signal_peak = 1000
        self._signal_width = 50
        self._pedestal_peak = 10
        self._pedestal_width = 40
        self._num_PMTs = 16

    def test_pedfitdim(self):
        """test pedestal fit return array for all 16 PMTs"""
        self.assertEqual(self._pedestals_dim, (self._num_PMTs, 6))

    def test_ped_fit_mean(self):
        """test pedestal fit peaks"""
        ped_peaks = self._pedestals[:, 0]
        ped_peaks_err = self._pedestals[:, 1]
        self._pedtest.plot_pedestals()
        for pmt in range(self._pedestals_dim[0]):
            lower_intervall = (
                self._pedestal_peak - 3.0 * ped_peaks_err[pmt]
            )  # 2*ped_peaks_err[pmt]
            upper_intervall = (
                self._pedestal_peak + 3.0 * ped_peaks_err[pmt]
            )  # 2*ped_peaks_err[pmt]
            print(
                self._pedestal_peak,
                ped_peaks[pmt],
                ped_peaks_err[pmt],
                lower_intervall,
                upper_intervall,
            )
            self.assertGreaterEqual(ped_peaks[pmt], lower_intervall)
            self.assertLessEqual(ped_peaks[pmt], upper_intervall)

    def test_ped_fit_width(self):
        """test pedestal fit widths"""
        ped_width = self._pedestals[:, 2]
        ped_width_err = self._pedestals[:, 3]
        for pmt in range(self._pedestals_dim[0]):
            lower_intervall = (
                self._pedestal_width - 3.0 * ped_width_err[pmt]
            )  # 3*ped_width_err[pmt]
            upper_intervall = (
                self._pedestal_width + 3.0 * ped_width_err[pmt]
            )  # 3*ped_width_err[pmt]
            self.assertGreaterEqual(ped_width[pmt], lower_intervall)
            self.assertLessEqual(ped_width[pmt], upper_intervall)

    def test_ped_correction(self):
        """test signal from pedestal correction"""
        measurement = MeasPerkeo(2, 5, list([self._rootfile]))
        custom_bins = {"bin_count": 500, "low_lim": 500, "up_lim": 2000}
        corr_class = CorrPerkeo(
            measurement,
            mode=2,
            ped_arr=self._pedestals,
            custom_pmt_hist_par=custom_bins,
        )
        corr_class.set_all_corr(bactive=False)
        corr_class.corrections["Pedestal"] = True
        corr_class.corr(bstore=True, bwrite=False)
        testhist = corr_class.histograms[0][1][0]
        # testhist.plot_hist()

        mu = self._signal_peak - self._pedestal_peak
        width = self._signal_width
        fitclass = DoFit(testhist.hist)
        fitclass.setup(gaus_simp)
        fitclass.limit_range([mu - 16 * width, mu + 16 * width])
        fitclass.set_bool("boutput", False)
        fitclass.set_fitparam("mu", mu)
        fitclass.set_fitparam("sig", width)
        fitclass.fit()

        fitted_mu = fitclass.ret_results().params["mu"].value
        fitted_mu_err = fitclass.ret_results().params["mu"].stderr
        print(fitted_mu, fitted_mu_err)
        lower_intervall = mu - 1.5 * fitted_mu_err
        upper_intervall = mu + 1.5 * fitted_mu_err
        self.assertGreaterEqual(fitted_mu, lower_intervall)
        self.assertLessEqual(fitted_mu, upper_intervall)
