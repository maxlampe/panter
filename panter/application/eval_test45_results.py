"""For plotting and evaluation ElecTest results from evalTest45.py"""

import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.config import conf_path
from panter import output_path
from lmfit import Model
from lmfit.model import ModelResult


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

outputFileDir = output_path + "/"

cnf2 = configparser.ConfigParser()
cnf2.read(f"{conf_path}/evalElec.ini")

TYPE = cnf2["studyRes"]["evalTest"]
BPLOT = cnf2["studyRes"].getboolean("BPLOT")
PNTS2SKP = int(cnf2["studyRes"]["points2skip"])

DPTT_A0 = 50
BUSE_ALT = False

BFIX_K_VAL = True
K_VAL = 99.0


def lin(x, a, b):
    """Simple, linear function in x."""
    return a * np.array(x, dtype=float) + b


def get_allparam(evalres: list, name: str) -> dict:
    """Get all of one parameter and its error over mult. Files from fit results."""

    values = [
        [
            file[dptt]["ModelRes"][ADC_IND].params[name].value
            if file[dptt]["ModelRes"][ADC_IND] is not None
            else None
            for dptt in dptt_values
        ]
        for file in evalres
    ]
    errors = [
        [
            file[dptt]["ModelRes"][ADC_IND].params[name].stderr
            if file[dptt]["ModelRes"][ADC_IND] is not None
            else None
            for dptt in dptt_values
        ]
        for file in evalres
    ]

    return {"val": values, "err": errors}


def get_allparam_t5(evalres: list, name: str) -> dict:
    """Get all of one parameter and its error over mult. Files from fit results."""

    values1 = [
        [file[dptt]["ModelRes1"][ADC_IND].params[name].value for dptt in dptt_values]
        for file in evalres
    ]
    values2 = [
        [file[dptt]["ModelRes2"][ADC_IND].params[name].value for dptt in dptt_values]
        for file in evalres
    ]
    errors1 = [
        [file[dptt]["ModelRes1"][ADC_IND].params[name].stderr for dptt in dptt_values]
        for file in evalres
    ]
    errors2 = [
        [file[dptt]["ModelRes2"][ADC_IND].params[name].stderr for dptt in dptt_values]
        for file in evalres
    ]
    return [{"val": values1, "err": errors1}, {"val": values2, "err": errors2}]


def combine_fitres(para: dict) -> list:
    """Combining parameter values for multiple files."""

    len_files = len(para["val"])
    len_dptt = len(para["val"][0])
    vals = np.zeros([len_dptt, 2])
    for i in range(0, len_dptt):
        sum_val = 0
        sum_err = 0
        for j in range(0, len_files):
            if para["err"][j][i] is not None:
                sum_val += para["val"][j][i] / para["err"][j][i] ** 2
                sum_err += 1 / para["err"][j][i] ** 2
        vals[i, 0] = sum_val / sum_err
        vals[i, 1] = np.sqrt(1 / sum_err)
    return np.array(vals).T


def ret_reldev(fit_result: ModelResult) -> list:
    """Calculate relative deviation of rate dependency."""

    c1_res = [fit_result.params["c1"].value, fit_result.params["c1"].stderr]
    a_res = [fit_result.params["a"].value, fit_result.params["a"].stderr]
    val = c1_res[0] / a_res[0]
    err = np.sqrt(
        (c1_res[1] / a_res[0]) ** 2 + (a_res[1] * c1_res[0] / (a_res[0]) ** 2) ** 2
    )

    return [val, err]


def ret_reldev_alt(fit_result: ModelResult) -> list:
    """Calculate relative deviation of rate dependency by using curve value."""

    a_res = [fit_result.params["a"].value, fit_result.params["a"].stderr]
    a_0 = [fit_result.eval(x=DPTT_A0), fit_result.eval_uncertainty(x=DPTT_A0)[0]]
    val = (a_res[0] - a_0[0]) / a_res[0]
    err = np.sqrt((a_res[1] * a_0[0] / (a_res[0]) ** 2) ** 2 + (a_0[1] / a_res[0]) ** 2)

    return [val, err]


def ret_reldev_t5(fit_result1: ModelResult, fit_result2: ModelResult) -> list:
    """Calculate relative deviation of rate dependency."""

    c1_res1 = [fit_result1.params["c1"].value, fit_result1.params["c1"].stderr]
    a_res1 = [fit_result1.params["a"].value, fit_result1.params["a"].stderr]
    c1_res2 = [fit_result2.params["c1"].value, fit_result2.params["c1"].stderr]
    a_res2 = [fit_result2.params["a"].value, fit_result2.params["a"].stderr]

    val1 = c1_res1[0] / a_res2[0]
    val2 = c1_res2[0] / a_res1[0]
    err1 = np.sqrt(
        (c1_res1[1] / a_res2[0]) ** 2 + (a_res2[1] * c1_res1[0] / (a_res2[0]) ** 2) ** 2
    )
    err2 = np.sqrt(
        (c1_res2[1] / a_res1[0]) ** 2 + (a_res1[1] * c1_res2[0] / (a_res1[0]) ** 2) ** 2
    )

    return [[val1, err1], [val2, err2]]


def ret_reldev_t5_alt(fit_result1: ModelResult, fit_result2: ModelResult) -> list:
    """Calculate relative deviation of rate dependency by using curve value."""

    a_res1 = [fit_result1.params["a"].value, fit_result1.params["a"].stderr]
    a_res2 = [fit_result2.params["a"].value, fit_result2.params["a"].stderr]
    a_0_1 = [fit_result1.eval(x=DPTT_A0), fit_result1.eval_uncertainty(x=DPTT_A0)[0]]
    a_0_2 = [fit_result2.eval(x=DPTT_A0), fit_result2.eval_uncertainty(x=DPTT_A0)[0]]

    val1 = (a_res1[0] - a_0_1[0]) / a_res2[0]
    val2 = (a_res2[0] - a_0_2[0]) / a_res1[0]
    err1 = np.sqrt(
        (a_res1[1] / a_res2[0]) ** 2
        + (a_0_1[1] / a_res2[0]) ** 2
        + (a_res2[1] * (a_res1[0] - a_0_1[0]) / (a_res2[0]) ** 2) ** 2
    )
    err2 = np.sqrt(
        (a_res2[1] / a_res1[0]) ** 2
        + (a_0_2[1] / a_res1[0]) ** 2
        + (a_res1[1] * (a_res2[0] - a_0_2[0]) / (a_res1[0]) ** 2) ** 2
    )

    return [[val1, err1], [val2, err2]]


EVAL_KEY = {
    "PMT0": [0, "Fan0", 0],
    "PMT1": [1, "Fan0", 1],
    "PMT2": [2, "Fan0", 2],
    "PMT3": [3, "Fan0", 3],
    "PMT4": [4, "Fan1", 0],
    "PMT5": [5, "Fan1", 1],
    "PMT6": [6, "Fan1", 2],
    "PMT7": [7, "Fan1", 3],
    "PMT8": [8, "Fan2", 0],
    "PMT9": [9, "Fan2", 1],
    "PMT10": [10, "Fan2", 2],
    "PMT11": [11, "Fan2", 3],
    "PMT12": [12, "Fan3", 0],
    "PMT13": [13, "Fan3", 1],
    "PMT14": [14, "Fan3", 2],
    "PMT15": [15, "Fan3", 3],
}

TEST4_FILES = {
    "Fan0": "Test4_0003_SG_bfixsig_2021-05-02.p",
    "Fan1": "Test4_0407_SG_bfixsig_2021-05-02.p",
    "Fan2": "Test4_0811_SG_bfixsig_2021-05-02.p",
    "Fan3": "Test4_1215_SG_bfixsig_2021-05-02.p",
}

TEST5_FILES = {
    "Fan0": "Test5_0003_SG_bfixsig_2021-05-03.p",
    "Fan1": "Test5_0407_SG_bfixsig_2021-05-03.p",
    "Fan2": "Test5_0811_SG_bfixsig_2021-05-03.p",
    "Fan3": "Test5_1215_SG_bfixsig_2021-05-03.p",
}

# ["050v300", "200v200", "200v500", "300v050", "300v300", "500v200", "500v500"]
EVAL_VALUES = [50.0, 200.0, 205.0, 300.0, 305.0, 500.0, 505.0]
EVAL_RESULTS = {
    "PMT0": [None] * 7,
    "PMT1": [None] * 7,
    "PMT2": [None] * 7,
    "PMT3": [None] * 7,
    "PMT4": [None] * 7,
    "PMT5": [None] * 7,
    "PMT6": [None] * 7,
    "PMT7": [None] * 7,
    "PMT8": [None] * 7,
    "PMT9": [None] * 7,
    "PMT10": [None] * 7,
    "PMT11": [None] * 7,
    "PMT12": [None] * 7,
    "PMT13": [None] * 7,
    "PMT14": [None] * 7,
    "PMT15": [None] * 7,
}

for pmt in EVAL_KEY:
    PMT_IND, FANOUT, ADC_IND = EVAL_KEY[pmt]
    print(f"Evaluation {pmt}: {FANOUT}, ADC {ADC_IND}")

    k_vals = []

    print("EVALUATE TEST 4")
    file_test4 = dP.FilePerkeo(outputFileDir + TEST4_FILES[FANOUT])
    imp_test4 = file_test4.imp()
    files_200mV, files_300mV, files_500mV = imp_test4

    dptt_values = files_200mV[0].index

    fitdata_test4 = {
        "200mV": combine_fitres(get_allparam(files_200mV, "mu")),
        "300mV": combine_fitres(get_allparam(files_300mV, "mu")),
        "500mV": combine_fitres(get_allparam(files_500mV, "mu")),
    }

    results = []
    for data in fitdata_test4:
        x_val = dptt_values[PNTS2SKP:]
        y_val = fitdata_test4[data][0][PNTS2SKP:]
        y_err = fitdata_test4[data][1][PNTS2SKP:]

        fitclass = eP.DoFit(pd.DataFrame({"x": x_val, "y": y_val, "err": y_err}))
        fitclass.setup(eFS.exp_sat_simp)
        if BFIX_K_VAL:
            fitclass.set_fitparam("k1", K_VAL, False)
        # fitclass.set_bool("boutput", True)
        fitclass.plot_labels = [f"{pmt}", "DPTT [mus]", "Peak pos [ch]"]
        fit_results = fitclass.fit()
        results.append(fit_results)
        k_vals.append(fit_results.params["k1"].value)

    if BUSE_ALT:
        EVAL_RESULTS[pmt][1] = ret_reldev_alt(results[0])
        EVAL_RESULTS[pmt][4] = ret_reldev_alt(results[1])
        EVAL_RESULTS[pmt][6] = ret_reldev_alt(results[2])
    else:
        EVAL_RESULTS[pmt][1] = ret_reldev(results[0])
        EVAL_RESULTS[pmt][4] = ret_reldev(results[1])
        EVAL_RESULTS[pmt][6] = ret_reldev(results[2])

    print("EVALUATE TEST 5")
    file_test5 = dP.FilePerkeo(outputFileDir + TEST5_FILES[FANOUT])
    imp_test5 = file_test5.imp()
    files_200500mV, files_050300mV = imp_test5

    dptt_values = files_200500mV[0].index

    get_allp_200500 = get_allparam_t5(files_200500mV, "mu")
    get_allp_050300 = get_allparam_t5(files_050300mV, "mu")

    fitdata_test5 = {
        "200500mV": combine_fitres(get_allp_200500[0]),
        "500200mV": combine_fitres(get_allp_200500[1]),
        "050300mV": combine_fitres(get_allp_050300[0]),
        "300050mV": combine_fitres(get_allp_050300[1]),
    }

    results = []
    for data in fitdata_test5:
        x_val = dptt_values[PNTS2SKP:]
        y_val = fitdata_test5[data][0][PNTS2SKP:]
        y_err = fitdata_test5[data][1][PNTS2SKP:]

        fitclass = eP.DoFit(pd.DataFrame({"x": x_val, "y": y_val, "err": y_err}))
        fitclass.setup(eFS.exp_sat_simp)
        if BFIX_K_VAL:
            fitclass.set_fitparam("k1", K_VAL, False)
        # fitclass.set_bool("boutput", True)
        fitclass.plot_labels = [f"{pmt}", "DPTT [mus]", "Peak pos [ch]"]
        fit_results = fitclass.fit()
        results.append(fit_results)
        k_vals.append(fit_results.params["k1"].value)

    if BUSE_ALT:
        EVAL_RESULTS[pmt][2], EVAL_RESULTS[pmt][5] = ret_reldev_t5_alt(
            results[0], results[1]
        )
        EVAL_RESULTS[pmt][0], EVAL_RESULTS[pmt][3] = ret_reldev_t5_alt(
            results[2], results[3]
        )
    else:
        EVAL_RESULTS[pmt][2], EVAL_RESULTS[pmt][5] = ret_reldev_t5(
            results[0], results[1]
        )
        EVAL_RESULTS[pmt][0], EVAL_RESULTS[pmt][3] = ret_reldev_t5(
            results[2], results[3]
        )
    print(np.array(k_vals).mean(), np.array(k_vals).std())

for pmt in EVAL_RESULTS:
    gmodel = Model(lin)
    params = gmodel.make_params(a=1.0, b=0.0)

    params["a"].vary = False
    params["a"].value = 0.0

    x_values = EVAL_VALUES
    y_values = np.array(EVAL_RESULTS[pmt]).T[0]
    yerr_values = np.array(EVAL_RESULTS[pmt]).T[1]

    df = pd.DataFrame({"x": x_values, "y": y_values, "err": yerr_values})
    df = dP.filt_zeros(df)

    fitres = gmodel.fit(
        df["y"],
        params,
        x=df["x"],
        weights=eP.calc_weights(df["err"]),
        scale_covar=False,
    )

    plt.errorbar(
        x_values,
        y_values,
        yerr=yerr_values,
        fmt=".",
    )
    plt.plot(
        x_values,
        gmodel.func(x_values, **fitres.params),
        "r-",
        label="best fit",
    )
    plt.annotate(
        f" f(x) = b \n"
        f'    b = {100*fitres.params["b"].value:.5f} +- '
        f'{100*fitres.params["b"].stderr:.5f} [%] \n'
        f" rChi2 = {fitres.redchi:0.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="1"),
    )
    plt.title(f"{pmt}")
    plt.xlabel("Amplitude [mV]")
    plt.ylabel("Rel dev [ ]")

    if BUSE_ALT:
        plt.savefig(f"reldev_a0ma_over_a_{pmt}.png")
    else:
        plt.savefig(f"reldev_c1_over_a_{pmt}.png")
    plt.clf()

    print(f'{pmt}:\t{fitres.params["b"].value}\t+- {fitres.params["b"].stderr}')
