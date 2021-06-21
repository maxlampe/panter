"""Module for evaluating imported Perkeo root data (RootPerkeo)."""

from __future__ import annotations

from datetime import datetime
from copy import deepcopy
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from lmfit import Model
from lmfit.model import ModelResult

import panter.core.dataPerkeo as dP
from panter.config import conf_path
import panter.config.evalFitSettings as eFS

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

# global plot output bool (default is False)
BPLOT = cnf["evalPerkeo"].getboolean("bplot")


def calc_weights(sample: np.array) -> np.array:
    """Function for calculating weights with 0 exception."""

    weights = []
    for ent in sample:
        assert ent != 0.0, "ERROR: Empty bin in weights calculation."
        weights.append(ent ** (-1))

    return np.array(weights)


class DoFit:
    """Class for executing fits on data for different models.

    This class replaces all fitFOO() functions. Fit settings need to be
    set/loaded though. No general defaults set, only model dependend
    defaults.

    Parameters
    ----------
    data : pd.DataFrame({'x': , 'y': , 'err': })
        Could be ret_hist() and therefore dP.HistPerkeo.hist

    Attributes
    ----------
    plotrange : [float, float]
    plot_labels : [str]*3
        (default ['Fit result', 'xLabel [ ]', 'yLabel [ ]'])
    blogscale : False
        Plot fitresults with y-axis log scaled.
    bfit_residuals : False
        Fit gaussian to fit residuals in DoFit.plot_fit()

    Examples
    --------
    General example of how to use DoFit. The histogram is imported from
    a RootPerkeo class object (dataSrc), which was generated outside of
    this example. The fit model is an exponentially modified Gaussian.

    >>> histogram = dataSrc.hists[0].hist
    >>> fitclass = eP.DoFit(histogram)
    >>> fitclass.setup(eFS.gaus_expmod)
    >>> fitclass.set_bool('boutput', True)
    >>> fitclass.fit()
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._fitdata = None
        self._label = None
        self._booldict = None
        self._fitrange = None
        self.plotrange = {"x": None, "y": None}
        self.plot_labels = ["Fit result", "xLabel [ ]", "yLabel [ ]"]
        self.blogscale = False
        self.bfit_residuals = False
        self._initvals = None
        self._fitfunc = None
        self._fitmodel = None
        self._fitparams = None
        self._fitresult = None
        self._brecparams = None
        self._gof = None

    def setup(self, fitsettings: eFS.FitSetting = eFS.gaus_simp):
        """Set attributes with FitSetting obj from evalFitSettings."""

        self._label = fitsettings.label
        self._booldict = fitsettings.booldict
        self._fitfunc = fitsettings.fitfunc
        self._initvals = fitsettings.initvals
        self._fitrange = fitsettings.fitrange
        self.plotrange = fitsettings.plotrange

        self._fitmodel = Model(self._fitfunc)
        self._fitparams = self._fitmodel.make_params(**self._initvals)

        for i, key in enumerate(self._fitparams):
            self._fitparams[key].vary = fitsettings.paramvary[i]

        if fitsettings.param_limit is not None:
            for key in fitsettings.param_limit:
                self._fitparams[key].min = fitsettings.param_limit[key][0]
                self._fitparams[key].max = fitsettings.param_limit[key][1]

        if fitsettings.plot_labels is not None:
            self.plot_labels = fitsettings.plot_labels

        return 0

    def limit_range(self, fitrange: [float, float]):
        """Activate and set fit range limit."""

        self._fitrange = fitrange
        self._booldict["blimfit"] = True

        return 0

    def set_bool(self, name: str, bvalue: bool):
        """set a chosen bool to desired value"""

        self._booldict[name] = bvalue

        return 0

    def set_fitparam(self, namekey: str, valpar: float = None, bparafree=None):
        """Set parameter with new value and/or set it free or not."""

        if valpar is None and bparafree is None:
            print('WARNING: Doing nothing. All inputs "None"!')
        if valpar is not None:
            self._fitparams[namekey].value = valpar
        if bparafree is not None:
            self._fitparams[namekey].vary = bparafree

        return 0

    def set_limit_fitparam(self, namekey: str, para_range: [float, float]):
        """Set range limit on fit parameter by keyword."""

        self._fitparams[namekey].min = para_range[0]
        self._fitparams[namekey].max = para_range[1]

        return 0

    def set_recursive(
        self, par_key_cen: str, par_key_wid: str, n_iter: int, factor: float
    ):
        """Set parameters for recursive fitting with range adaption."""

        self._brecparams = {
            "par_key_cen": par_key_cen,
            "par_key_wid": par_key_wid,
            "n_iter": n_iter,
            "factor": factor,
        }
        self._booldict["brecfit"] = True

        return 0

    def fit(self) -> ModelResult:
        """Do fit with current settings."""

        if self._booldict["blimfit"]:
            self._fitdata = self._data.query(
                f"{self._fitrange[0]}" f"< x <" f"{self._fitrange[1]}"
            )
        else:
            self._fitdata = self._data

        # check for label and then do case specific range limitation
        if self._label in ["gaus_pdec", "gaus_expmod", "charge_spec"]:
            maxpos = np.argmax(self._fitdata["y"])
            peakpos = self._fitdata["x"].values[maxpos]

            if self._label == "gaus_pdec":
                self._fitparams["c3"].value = peakpos
            elif self._label == "gaus_expmod":
                self._fitparams["mu"].value = peakpos

            self._fitrange = [peakpos * 0.7 - 150.0, peakpos * 1.4 + 300.0]
            self._fitdata = self._data.query(
                f"{self._fitrange[0]}" f" < x < " f"{self._fitrange[1]}"
            )

        if self._label == "gaus_gen":
            self.set_recursive("mu", "sig", 2, 1.35)

        self._fitdata = dP.filt_zeros(self._fitdata)
        err_weights = calc_weights(self._fitdata["err"])

        if len(self._fitparams) <= self._fitdata["y"].shape[0]:
            self._fitresult = self._fitmodel.fit(
                self._fitdata["y"],
                self._fitparams,
                x=self._fitdata["x"],
                weights=err_weights,
                scale_covar=False,
            )
        else:
            self._fitresult = None

        if self._booldict["brecfit"]:
            key_center = self._brecparams["par_key_cen"]
            key_width = self._brecparams["par_key_wid"]
            fac = self._brecparams["factor"]

            for _ in range(0, self._brecparams["n_iter"]):
                delta = fac * abs(self._fitresult.params[key_width].value)
                self._fitdata = self._data.query(
                    f"{self._fitresult.params[key_center].value - delta}"
                    f" < x < "
                    f"{self._fitresult.params[key_center].value + delta}"
                )
                self._fitdata = dP.filt_zeros(self._fitdata)

            err_weights = calc_weights(self._fitdata["err"])
            if len(self._fitparams) <= self._fitdata["y"].shape[0]:
                self._fitresult = self._fitmodel.fit(
                    self._fitdata["y"],
                    self._fitparams,
                    x=self._fitdata["x"],
                    weights=err_weights,
                )
            else:
                self._fitresult = None

        if self._fitresult is not None:
            pval = 1.0 - chi2.cdf(self._fitresult.chisqr, self._fitresult.nfree)
            self._gof = [self._fitresult.redchi, pval]
        else:
            self._gof = [None, None]

        if self._booldict["boutput"]:
            self.plot_fit()

        return self._fitresult

    def print_geninfo(self):
        """Print general infos about fit settings."""

        print("\n--- Printing fit settings info\n")
        print(f"\tFit model:\t{self._fitmodel}")
        print(f"\tFit parameter names:\n\t{self._fitmodel.param_names}")
        print(f"\tIndep. parameter(s):\n\t{self._fitmodel.independent_vars}")
        print(f"\tInit vals = {self._initvals}")
        print(f"\tFit range = {self._fitrange}")
        print(f"\tPlot range = {self.plotrange}")
        print(f"\tBools:\n\t{self._booldict}")
        print(f"\tRecursive fit parameters:\n\t{self._brecparams}")

        return 0

    def print_fitinfo(self):
        """Print fit results."""

        print("\n--- Printing fit info\n")
        print(self._fitresult.fit_report())
        print("p value = \t", self._gof[1])

        return 0

    def ret_results(self):
        """Return fit results."""

        if self._fitresult is None:
            print('WARNING: No fit results yet set. Returning "None".')
        return self._fitresult

    def ret_gof(self):
        """Return goodness of fit results."""

        if self._gof is None:
            print('WARNING: No fit results yet set. Returning "None".')
        return self._gof

    def ret_fitrange(self):
        """Return goodness of fit results."""

        if self._fitrange is None:
            print('WARNING: No fit range yet set. Returning "None".')
        return self._fitrange

    def plot_fit(self):
        """Plot data, fitresult and residuals of current fit."""

        self.print_fitinfo()
        residuals_data = self._fitdata["y"] - self._fitmodel.eval(
            **self._fitresult.params, x=self._fitdata["x"]
        )
        all_residuals = self._data["y"] - self._fitmodel.eval(
            **self._fitresult.params, x=self._data["x"]
        )

        residual_hist = dP.HistPerkeo(
            residuals_data,
            int(len(residuals_data) * 0.5),
            residuals_data.min() * 0.91,
            residuals_data.max() * 1.1,
        )

        if self.bfit_residuals:
            fitclass = DoFit(residual_hist.hist)
            fitclass.setup(eFS.gaus_simp)
            fitclass.set_fitparam("mu", valpar=0.0)
            fitclass.set_fitparam("sig", valpar=3.0)
            fitclass.set_fitparam("norm", valpar=20.0)
            residual_gausfit = fitclass.fit()

        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [6, 2]},
        )
        fig.subplots_adjust(hspace=0)
        axs[0].set_title(self.plot_labels[0])

        axs.flat[0].set(ylabel=self.plot_labels[2])
        if self.plotrange["x"] is not None:
            axs[0].set_xlim(self.plotrange["x"])
        if self.plotrange["y"] is not None:
            axs[0].set_ylim(self.plotrange["y"])
        if self.blogscale:
            axs[0].set_yscale("log")

        axs[0].errorbar(self._data["x"], self._data["y"], self._data["err"], fmt=".")

        axs[0].plot(
            self._data["x"],
            self._fitmodel.eval(**self._fitresult.params, x=self._data["x"]),
            "r-",
            label="best fit",
        )

        axs[0].grid(True)
        axs[0].annotate(
            f"{self._fitmodel.name} \n"
            f"rCh2 = {self._gof[0]:.3f}\n"
            f"p = {self._gof[1]:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="1"),
        )

        axs.flat[1].set(xlabel=self.plot_labels[1], ylabel="Residuals [x]")
        axs[1].errorbar(
            self._data["x"],
            all_residuals,
            self._data["err"],
            fmt=".",
        )
        axs[1].errorbar(
            self._fitdata["x"],
            residuals_data,
            self._fitdata["err"],
            fmt=".",
            color="r",
            label="fit range",
        )
        axs[1].legend()

        axs[1].grid(True)
        if self.bfit_residuals:
            try:
                axs[1].annotate(
                    f'Res gaus Fit: mu = {residual_gausfit.params["mu"].value:.2f} +- '
                    f'{residual_gausfit.params["mu"].stderr:.2f}  '
                    f'sig = {residual_gausfit.params["sig"].value:.2f}'
                    f'+- {residual_gausfit.params["sig"].stderr:.2f}   '
                    f"rChi = {residual_gausfit.redchi:.2f} ",
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    bbox=dict(boxstyle="round", fc="1"),
                )
            except TypeError:
                print("WARNING: Residual Gaussian fit failed.")
        plt.show()

        return 0

    def scanfit(self, param: str):
        """UNFINISHED - WIP"""
        # scan one fit parameter and study chi**2 fresults
        pass

    def compfit(self, otherfit: DoFit):
        """UNFINISHED - WIP"""
        # comapre fit models by plotting / calculating difference etc
        pass


class DoFitData:
    """Class for fitting data from certain measurements.

    Fit settings cannot be set/loaded though as settings are set by
    measurement type and ini file parameters. Logging available.

    Parameters
    ----------
    dataclass : dP.RootPerkeo
    datatype : {'DriftSn', 'ElecTest4', 'ElecTest5',
                'ElecTest4_fixSig', 'ElecTest5_fixSig'}
        Implemented measurement evaluation settings

    Examples
    --------
    General example of how to use DoFitData. The data is imported as as
    RootPerkeo class object (here dataSrc), which was generated outside
    of this example. The fit model is an exponentially modified Gaussian.

    >>> fitdataclass = eP.DoFitData(dataSrc, 'DriftSn')
    >>> fit_result = fitdataclass.fit()
    >>> fitdataclass.write_2log(outputFileDir + LOGFILE)
    """

    def __init__(self, dataclass: dP.RootPerkeo, datatype: str):
        self._dataclass = dataclass
        self._datatype = datatype
        self._fitsettings = None
        self._valid_datatypes = [
            "DriftSn",
            "ElecTest4",
            "ElecTest5",
            "ElecTest4_fixSig",
            "ElecTest5_fixSig",
        ]
        self.boutput = BPLOT
        self._stats = {
            "nPMT": None,
            "dataPMT": None,
            "nMean": None,
            "nSig": None,
            "nSig2D": None,
            "currHisP": None,
        }

    def set_stats(self):
        """Calculate various stats from data for fits."""

        if not self._fitsettings.booldict["bfitall"]:
            self._stats["nPMT"] = self._dataclass.ret_actpmt()
        else:
            self._stats["nPMT"] = list(range(0, self._dataclass.no_pmts))
        self._stats["dataPMT"] = self._dataclass.pmt_data
        self._stats["nMean"] = self._dataclass.stats["mean"]
        self._stats["nSig"] = self._dataclass.stats["sig"]
        self._stats["nSig2D"] = self._dataclass.stats["sig2D"]
        self._stats["currHisP"] = self._dataclass.hists

        return 0

    def fit(self) -> list:
        """Depending on set datatype, fit will be done."""

        # check for valid fitmodel input
        assert (
            np.array(self._valid_datatypes) == self._datatype
        ).sum() > 0, "ERROR: Wrong fitmodel/datatype input!"

        if self._datatype == self._valid_datatypes[0]:
            self._fitsettings = eFS.gaus_expmod
            self._fitsettings.plot_labels = [
                "SnSpec fit result",
                "ADC [ch]",
                "Counts [ ]",
            ]
            self._fitsettings.plotrange["x"] = [30.0, 4000.0]
            self.set_stats()
        if (
            self._datatype == self._valid_datatypes[1]
            or self._datatype == self._valid_datatypes[3]
        ):
            self._fitsettings = eFS.gaus_gen
            self._fitsettings.plot_labels = [
                "Gaussian fit result",
                "ADC [ch]",
                "Counts [ ]",
            ]
            self.set_stats()
        if (
            self._datatype == self._valid_datatypes[2]
            or self._datatype == self._valid_datatypes[4]
        ):
            self._fitsettings = eFS.gaus_gen
            self._fitsettings.plot_labels = [
                "Gaussian fit result",
                "ADC [ch]",
                "Counts [ ]",
            ]
            # create 2D for second Gaussian
            self.set_stats()
            self._fitsettings = [
                self._fitsettings,
                deepcopy(self._fitsettings),
            ]

        fitresults = []
        for i in self._stats["nPMT"]:

            if (
                self._datatype == self._valid_datatypes[1]
                or self._datatype == self._valid_datatypes[3]
            ):
                self._fitsettings.initvals["mu"] = self._stats["nMean"][i]
                self._fitsettings.initvals["sig"] = self._stats["nSig"][i]

                self._fitsettings.plotrange = [
                    self._stats["nMean"][i] - 4 * self._stats["nSig"][i],
                    self._stats["nMean"][i] + 4 * self._stats["nSig"][i],
                ]

            if (
                self._datatype == self._valid_datatypes[2]
                or self._datatype == self._valid_datatypes[4]
            ):
                self._fitsettings[0].fitrange = [0.0, self._stats["nMean"][i]]
                self._fitsettings[1].fitrange = [
                    self._stats["nMean"][i],
                    16000.0,
                ]

                self._fitsettings[0].initvals["mu"] = (
                    self._stats["nMean"][i] - self._stats["nSig"][i]
                )
                self._fitsettings[1].initvals["mu"] = (
                    self._stats["nMean"][i] + self._stats["nSig"][i]
                )
                if self._datatype == self._valid_datatypes[4]:
                    self._fitsettings[0].initvals["sig"] = self._stats["nSig2D"][i][0]
                    self._fitsettings[1].initvals["sig"] = self._stats["nSig2D"][i][1]
                else:
                    self._fitsettings[0].initvals["sig"] = 0.1 * self._stats["nSig"][i]
                    self._fitsettings[1].initvals["sig"] = 0.1 * self._stats["nSig"][i]
                dofitclass1 = DoFit(self._dataclass.hists[i].hist)
                dofitclass1.setup(self._fitsettings[0])
                dofitclass1.set_bool("boutput", self.boutput)
                if self._datatype == self._valid_datatypes[4]:
                    dofitclass1.set_fitparam(namekey="sig", bparafree=False)
                dofitclass1.fit()

                dofitclass2 = DoFit(self._dataclass.hists[i].hist)
                dofitclass2.setup(self._fitsettings[1])
                dofitclass2.set_bool("boutput", self.boutput)
                if self._datatype == self._valid_datatypes[4]:
                    dofitclass2.set_fitparam(namekey="sig", bparafree=False)
                dofitclass2.fit()

                fitresults.append(
                    [
                        i,
                        dofitclass1.ret_gof(),
                        dofitclass2.ret_gof(),
                        dofitclass1.ret_results(),
                        dofitclass2.ret_results(),
                    ]
                )
            else:
                dofitclass = DoFit(self._dataclass.hists[i].hist)
                dofitclass.setup(self._fitsettings)
                dofitclass.set_bool("boutput", self.boutput)
                if self._datatype == self._valid_datatypes[3]:
                    dofitclass.set_fitparam(namekey="sig", bparafree=False)
                dofitclass.fit()

                fitresults.append([i, dofitclass.ret_gof(), dofitclass.ret_results()])

        fitresults = np.array(fitresults)

        return fitresults

    def write_2log(self, logfilename: str):
        """Write current evaluation settings to a Log file (appending)"""

        currt = datetime.now()
        set1 = (
            cnf["dataPerkeo"]["ADC_hist_counts"]
            + "\t"
            + cnf["dataPerkeo"]["ADC_hist_min"]
            + "\t"
            + cnf["dataPerkeo"]["ADC_hist_max"]
            + "\t"
        )

        cnf_int = configparser.ConfigParser()
        if (self._datatype == np.array(self._valid_datatypes)[1:]).sum() == 1:
            cnf_int.read(f"{conf_path}/evalElec.ini")
            set2 = (
                cnf_int["evalFits"]["Eval_DPTT_Delta"]
                + "\t"
                + cnf_int["evalFits"]["Eval_DPTT_imin"]
                + "\t"
                + cnf_int["evalFits"]["Eval_DPTT_iMax"]
                + "\t"
                + cnf_int["evalFits"]["Eval_DPTT_iSig"]
                + "\t"
                + cnf_int["evalFits"]["BSigFix"]
                + "\t"
                + cnf_int["evalFits"]["BNorFix"]
                + "\t"
                + cnf_int["evalFits"]["BFitAll"]
            )

        if self._datatype == self._valid_datatypes[0]:
            cnf_int.read("evalDrift.ini")
            set2 = cnf_int["evalFits"]["BFitAll"]

        settings = str(currt) + "\t" + self._datatype + "\t" + set1 + set2 + "\n"

        del cnf_int
        file = dP.FilePerkeo(logfilename)
        print("Write to log file", file.dump(obj=str(settings), bapp=True, btext=True))

        return 0


def scan_fit(param, mod, fres, fdat, bplot, brefit=False):
    # simple routine to take a finished fit, fix all parameters and vary one
    # around its result to return Chi2 and optionally plot it as well

    param_vals = np.linspace(
        fres.params[param].value - 4.0 * fres.params[param].stderr,
        fres.params[param].value + 4.0 * fres.params[param].stderr,
        1000,
    )
    scanres = []
    params = mod.make_params(**fres.params)
    if not brefit:
        params.vary = False
    else:
        params[param].vary = False

    for i in param_vals:
        params[param].value = i
        res = mod.fit(fdat["y"], params, x=fdat["x"], weights=calc_weights(fdat["err"]))
        scanres.append([i, res.chisqr])
    scanres = np.array(scanres)

    if bplot:
        axes = plt.gca()
        axes.set_xlabel(param + " [ ]")
        axes.set_ylabel("Chi^2 [ ]")
        axes.set_xlim([scanres[:, 0].min() * 0.99, scanres[:, 0].max() * 1.02])
        axes.set_ylim(
            [
                2 * scanres[:, 1].min() - scanres[:, 1].max(),
                2 * scanres[:, 1].max() - scanres[:, 1].min(),
            ]
        )
        axes.grid(True)
        plt.plot(scanres[:, 0], scanres[:, 1], ".", label="Scan data")
        plt.axvline(fres.params["norm"].value, -1.0, 1.0, label="Fit result")
        plt.legend(loc="best")
        plt.title("Chi2 scan result")
        plt.tight_layout()
        plt.show()

    return scanres


class PedPerkeo:
    """
    Class for generating, plotting and storing pedestal values of PERKEO III datasets.

    Does not overwrite previous filters in dataclass dP.RootPerkeo.

    Parameters
    ----------
    dataclass : dP.RootPerkeo
    bplot_res, bplot_fit, bplot_log: False, False, False
        Activate plotting the pedestal results, plotting each fit result and plotting
        each fit result with a log scaled y-axis.
    bnaive_filt: False
        If True, deactivates standard filter (detector == 0/1) to get electronic Eigen-
        signal and fit a Gaussian to data below 500 channels for each PMT. Useful for
        e.g. electronics data with non-organic trigger patterns.
    range_detsum: None
        Filter DetSum for specific range (list). If None, it's not applied.
    range_dtt: None
        Filter DeltraTriggerTime for specific range (list). If None, it's not applied.

    Examples
    --------
    >>> data = dP.RootPerkeo(filename)
    >>> pedtest = eP.PedPerkeo(data)
    >>> pedtest.plot_pedestals()
    >>> print(pedtest.ret_pedestals())
    """

    def __init__(
        self,
        dataclass: dP.RootPerkeo,
        bplot_res: bool = False,
        bplot_fit: bool = False,
        bplot_log: bool = False,
        bnaive_filt: bool = False,
        range_detsum: list = None,
        range_dtt: list = None,
    ):
        self._dataclass = dataclass
        self._bplot_fit = bplot_fit
        self._bplot_log = bplot_log
        self._bnaive_filt = bnaive_filt
        self._range_detsum = range_detsum
        self._range_dtt = range_dtt

        if self._dataclass.no_pmts is None:
            self._dataclass.auto()
        self._ped_hist_par = {
            "bin_count": int(cnf["dataPerkeo"]["PED_hist_counts"]),
            "low_lim": int(cnf["dataPerkeo"]["PED_hist_min"]),
            "up_lim": int(cnf["dataPerkeo"]["PED_hist_max"]),
        }
        self._pedvalues = np.asarray(self.calc_ped())
        self.ped_hists = None
        if bplot_res:
            self.plot_pedestals()

    def calc_ped(self):
        """Calculating pedestals"""

        ped_list = [None] * self._dataclass.no_pmts
        ped_hists = np.asarray([None] * self._dataclass.no_pmts)

        for DET in [0, 1]:
            self._dataclass.clear_filt()
            if not self._bnaive_filt:
                self._dataclass.set_filt(
                    "data", fkey="Detector", active=True, ftype="bool", rightval=1 - DET
                )

            if self._range_detsum is not None:
                self._dataclass.set_filt(
                    "data",
                    fkey="DetSum",
                    active=True,
                    ftype="num",
                    low_lim=self._range_detsum[0],
                    up_lim=self._range_detsum[1],
                )
            if self._range_dtt is not None:
                self._dataclass.set_filt(
                    "data",
                    fkey="DeltaTriggerTime",
                    active=True,
                    ftype="num",
                    low_lim=self._range_dtt[0],
                    up_lim=self._range_dtt[1],
                )
            self._dataclass.auto(1)
            if DET == 0:
                for i in range(0, 8):
                    ped_hists[i] = dP.HistPerkeo(
                        self._dataclass.pmt_data[i], **self._ped_hist_par
                    )

            elif DET == 1:
                for i in range(8, 16):
                    ped_hists[i] = dP.HistPerkeo(
                        self._dataclass.pmt_data[i], **self._ped_hist_par
                    )

        for ind_hist, hist in enumerate(ped_hists):
            if hist is not None:
                histogram = hist.hist

                for i in [0, 1]:
                    fitclass = DoFit(histogram)
                    fitclass.setup(eFS.gaus_simp)
                    # FIXME: Why is this necessary?
                    # FIXME: Why does it change eFS.gaus_simp instead of fitclass att?
                    fitclass.set_bool("blimfit", False)
                    fitclass.set_bool("boutput", False)

                    if i == 0:
                        fitclass.set_fitparam(namekey="mu", valpar=0.0)
                        if self._bnaive_filt:
                            fitclass.limit_range([-np.inf, 500.0])
                        fitclass.fit()

                        if fitclass.ret_results() is not None:
                            mu_start = fitclass.ret_results().params["mu"].value
                            sig_start = np.abs(
                                fitclass.ret_results().params["sig"].value
                            )
                            fit_range = [
                                mu_start - 1.1 * sig_start,
                                mu_start + 0.5 * sig_start,
                            ]
                        else:
                            break

                    else:
                        fitclass.set_fitparam(namekey="mu", valpar=mu_start)
                        fitclass.set_fitparam(namekey="sig", valpar=sig_start)
                        fitclass.limit_range(fit_range)
                        if self._bplot_fit:
                            if self._bplot_log:
                                fitclass.blogscale = self._bplot_log
                                plot_yr = [0.1, histogram["y"].max() * 10.0]
                                fitclass.plotrange["y"] = plot_yr
                            fitclass.set_bool("boutput", True)
                            plot_xr = [
                                mu_start - 7.0 * sig_start,
                                mu_start + 7.0 * sig_start,
                            ]
                            fitclass.plotrange["x"] = plot_xr
                        fitclass.fit()

                if fitclass.ret_results() is not None:
                    ped_list[ind_hist] = [
                        fitclass.ret_results().params["mu"].value,
                        fitclass.ret_results().params["mu"].stderr,
                        np.abs(fitclass.ret_results().params["sig"].value),
                        fitclass.ret_results().params["sig"].stderr,
                        fitclass.ret_gof()[0],
                        fitclass.ret_gof()[1],
                    ]

        self.ped_hists = ped_hists
        self._dataclass.clear_filt()
        # TODO: there was an issue here. see pedestal_analysis
        # print(self.ped_hists)

        return ped_list

    def ret_pedestals(self) -> np.array(float):
        return self._pedvalues

    def plot_pedestals(self):
        """Plotting pedestal fit results."""

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

        axs[0].set_title("Pedestal Gaussian Mu w Err")
        axs.flat[0].set(ylabel="Ped Mu [ch]")
        axs[0].errorbar(
            x=range(0, 16),
            y=self._pedvalues.T[0],
            yerr=self._pedvalues.T[1],
            fmt=".",
        )

        axs[1].set_title("Pedestal Gaussian Sig w Err")
        axs.flat[1].set(ylabel="Ped Sig [ch]")
        axs[1].errorbar(
            x=range(0, 16),
            y=self._pedvalues.T[2],
            yerr=self._pedvalues.T[3],
            fmt=".",
        )
        plt.show()

        return 0
