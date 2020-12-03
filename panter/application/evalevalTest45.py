"""For plotting and evaluation ElecTest results from evalTest45.py"""

import configparser
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read("evalRaw.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]

cnf2 = configparser.ConfigParser()
cnf2.read("evalElec.ini")

TYPE = cnf2["studyRes"]["evalTest"]
bplot = cnf2["studyRes"].getboolean("BPLOT2")


def print_fitrestable(fitreslist):
    """Helper function: Take fitresults and print them."""

    print(
        "{:7s} {:14s} {:18s} {:12s}".format("PMT", "a", "+- err[ch]", "k1"),
        "{:19s} {:12s} {:18s}".format("+- err[muHz]", "c1", "+- err[ch]"),
        "{:14s} {:14s}".format("t [mus]", "a@t [ch]"),
        "{:15s} {:11s}".format("redChi", "p val"),
    )

    ks1, cs1, ass1 = [], [], []

    for i, i_val in enumerate(fitreslist):
        print("%i\t" % i, end="")
        for j_val in i_val:
            print("{:7.2f} \t".format(j_val), end="")
        print("")

        ass1.append(fitreslist[i][0])
        ks1.append(fitreslist[i][2])
        cs1.append(fitreslist[i][4])

    print("Mean a =\t%0.2f" % np.array(ass1).mean())
    print("Mean k =\t%0.2f" % np.array(ks1).mean())
    print("Mean c =\t%0.2f" % np.array(cs1).mean())

    return 0


# Import fit results for test
if TYPE == "Test4":

    file = dP.FilePerkeo(outputFileDir + "Test4_0003_SG_FixSig_2020-08-31.p")
    imp = file.imp()
    # should be the same for all cases
    currFilt = np.array(imp[0][0])[:, 0] * 0.01  # [mus]
    # TODO: fileinlist iterator fix for first file --> not using all data atm
    # [sublist][fileinlist][y val (Evalpoint)]
    Test4_0003_200 = np.array(imp[0][0])[:, 1]
    Test4_0003_300 = np.array(imp[1][0])[:, 1]
    Test4_0003_500 = np.array(imp[2][0])[:, 1]

    file = dP.FilePerkeo(outputFileDir + "Test4_0407_SG_FixSig_2020-08-31.p")
    imp = file.imp()
    Test4_0407_200 = np.array(imp[0][0])[:, 1]
    Test4_0407_300 = np.array(imp[1][0])[:, 1]
    Test4_0407_500 = np.array(imp[2][0])[:, 1]

    file = dP.FilePerkeo(outputFileDir + "Test4_0811_SG_FixSig_2020-08-31.p")
    imp = file.imp()
    Test4_0811_200 = np.array(imp[0][0])[:, 1]
    Test4_0811_300 = np.array(imp[1][0])[:, 1]
    Test4_0811_500 = np.array(imp[2][0])[:, 1]

    file = dP.FilePerkeo(outputFileDir + "Test4_1215_SG_FixSig_2020-08-31.p")
    imp = file.imp()
    Test4_1215_200 = np.array(imp[0][0])[:, 1]
    Test4_1215_300 = np.array(imp[1][0])[:, 1]
    Test4_1215_500 = np.array(imp[2][0])[:, 1]

else:
    if TYPE == "Test5":
        file = dP.FilePerkeo(outputFileDir + "Test5_0003_SG_FixSig_2020-08-28.p")
        imp = file.imp()
        # should be the same for all cases
        currFilt = np.array(imp[0][0])[:, 0] * 0.01  # [mus]
        # TODO: fileinlist iterator fix for first file --> not using all data!
        Test5_0003_200500 = np.array(imp[0][0])[:, 1]
        Test5_0003_050300 = np.array(imp[1][0])[:, 1]

        file = dP.FilePerkeo(outputFileDir + "Test5_0407_SG_FixSig_2020-08-28.p")
        imp = file.imp()
        Test5_0407_200500 = np.array(imp[0][0])[:, 1]
        Test5_0407_050300 = np.array(imp[1][0])[:, 1]

        file = dP.FilePerkeo(outputFileDir + "Test5_0811_SG_FixSig_2020-08-28.p")
        imp = file.imp()
        Test5_0811_200500 = np.array(imp[0][0])[:, 1]
        Test5_0811_050300 = np.array(imp[1][0])[:, 1]

        file = dP.FilePerkeo(outputFileDir + "Test5_1215_SG_FixSig_2020-08-28.p")
        imp = file.imp()
        Test5_1215_200500 = np.array(imp[0][0])[:, 1]
        Test5_1215_050300 = np.array(imp[1][0])[:, 1]

    else:
        print("ERROR: Unavailable Test to evaluate! \t", TYPE)
        sys.exit(1)


# Eval of fit results

PMTSig1, PMTSig2, PMTSig3, PMTSig4 = np.arange(0, 4)
PMTSig1All, PMTSig2All, PMTSig3All, PMTSig4All = [], [], [], []

if TYPE == "Test4":
    databyPMT = {
        "PMT0003": [Test4_0003_200, Test4_0003_300, Test4_0003_500],
        "PMT0407": [Test4_0407_200, Test4_0407_300, Test4_0407_500],
        "PMT0811": [Test4_0811_200, Test4_0811_300, Test4_0811_500],
        "PMT1215": [Test4_1215_200, Test4_1215_300, Test4_1215_500],
    }
    databyAmp = {
        "200mV": [Test4_0003_200, Test4_0407_200, Test4_0811_200, Test4_1215_200],
        "300mV": [Test4_0003_300, Test4_0407_300, Test4_0811_300, Test4_1215_300],
        "500mV": [Test4_0003_500, Test4_0407_500, Test4_0811_500, Test4_1215_500],
    }

    # TAG = 'PMT0003'
    # dataall = databyPMT[TAG]
    TAG = "200mV"
    dataall = databyAmp[TAG]

if TYPE == "Test5":
    databyPMT = {
        "PMT0003": [Test5_0003_200500, Test5_0003_050300],
        "PMT0407": [Test5_0407_200500, Test5_0407_050300],
        "PMT0811": [Test5_0811_200500, Test5_0811_050300],
        "PMT1215": [Test5_1215_200500, Test5_1215_050300],
    }
    databyAmp = {
        "200500mV": [
            Test5_0003_200500,
            Test5_0407_200500,
            Test5_0811_200500,
            Test5_1215_200500,
        ],
        "050300mV": [
            Test5_0003_050300,
            Test5_0407_050300,
            Test5_0811_050300,
            Test5_1215_050300,
        ],
    }

    # TAG = 'PMT0003'
    # dataall = databyPMT[TAG]
    TAG = "050300mV"
    dataall = databyAmp[TAG]

fitres = []
if TYPE == "Test5":
    fitres2 = []
for j in dataall:
    PMTSig1All, PMTSig2All, PMTSig3All, PMTSig4All = [], [], [], []

    for i in j:
        PMTSig1All.append(i[PMTSig1])
        PMTSig2All.append(i[PMTSig2])
        PMTSig3All.append(i[PMTSig3])
        PMTSig4All.append(i[PMTSig4])

    PMTSig1All, PMTSig2All, PMTSig3All, PMTSig4All = (
        np.asarray(PMTSig1All).T,
        np.asarray(PMTSig2All).T,
        np.asarray(PMTSig3All).T,
        np.asarray(PMTSig4All).T,
    )

    plotdata = {
        "PMT0": PMTSig1All,
        "PMT1": PMTSig2All,
        "PMT2": PMTSig3All,
        "PMT3": PMTSig4All,
    }

    # start data point to skip 'weird' first two-four points
    startData = int(cnf2["studyRes"]["points2skip"])

    for i in plotdata:
        res1, fig1 = eP.fitexp_sat(
            {
                "x": currFilt[startData:],
                "y": plotdata[i][1][startData:],
                "err": plotdata[i][2][startData:],
            },
            {"a": 5800.0, "k1": 100.0, "c1": 10.0},
            bplot,
            0,
        )
        if TYPE == "Test5":
            res2, fig2 = eP.fitexp_sat(
                {
                    "x": currFilt[startData:],
                    "y": plotdata[i][7][startData:],
                    "err": plotdata[i][8][startData:],
                },
                {"a": 5800.0, "k1": 100.0, "c1": 10.0},
                bplot,
                0,
            )
        if bplot:
            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(15, 15))
            fig.suptitle("%s : %s %s Result" % (i, TYPE, TAG))
            axs[0, 0].set_title("Gaussian Mu1")
            axs[0, 0].errorbar(
                currFilt[startData:],
                plotdata[i][1][startData:],
                plotdata[i][2][startData:],
                fmt=".",
            )
            axs[0, 0].plot(currFilt[startData:], res1.best_fit, "r-", label="best fit")
            if TYPE == "Test5":
                axs[0, 1].set_title("Gaussian Mu2")
                axs[0, 1].errorbar(
                    currFilt[startData:],
                    plotdata[i][7][startData:],
                    plotdata[i][8][startData:],
                    fmt=".",
                )
                axs[0, 1].plot(
                    currFilt[startData:], res2.best_fit, "r-", label="best fit"
                )

            axs[1, 0].set_title("Gaussian sig")
            axs.flat[2].set(xlabel="Time to prev. trigger [mus]", ylabel="Sigma [ch]")
            axs[1, 0].errorbar(
                currFilt[startData:],
                abs(plotdata[i][3][startData:]),
                plotdata[i][4][startData:],
                fmt=".",
            )
            if TYPE == "Test5":
                axs[1, 0].errorbar(
                    currFilt[startData:],
                    abs(plotdata[i][9][startData:]),
                    plotdata[i][10][startData:],
                    fmt=".",
                )

            axs[1, 1].set_title("Gaussian norms")
            axs.flat[3].set(xlabel="Time to prev. trigger [mus]", ylabel="Norm [ch]")
            axs[1, 1].errorbar(
                currFilt[startData:],
                plotdata[i][5][startData:],
                plotdata[i][6][startData:],
                fmt=".",
            )
            if TYPE == "Test5":
                axs[1, 1].errorbar(
                    currFilt[startData:],
                    plotdata[i][11][startData:],
                    plotdata[i][12][startData:],
                    fmt=".",
                )
            plt.show()

        fitres.append(
            np.array(
                [
                    res1.params["a"].value,
                    res1.params["a"].stderr,
                    res1.params["k1"].value,
                    res1.params["k1"].stderr,
                    res1.params["c1"].value,
                    res1.params["c1"].stderr,
                    currFilt[startData + 5],
                    res1.best_fit[5],
                    res1.redchi,
                    1.0 - chi2.cdf(res1.chisqr, res1.nfree),
                ]
            )
        )
        if TYPE == "Test5":
            fitres2.append(
                np.array(
                    [
                        res2.params["a"].value,
                        res2.params["a"].stderr,
                        res2.params["k1"].value,
                        res2.params["k1"].stderr,
                        res2.params["c1"].value,
                        res2.params["c1"].stderr,
                        currFilt[startData + 5],
                        res2.best_fit[5],
                        res2.redchi,
                        1.0 - chi2.cdf(res2.chisqr, res2.nfree),
                    ]
                )
            )

if TYPE == "Test4":
    print_fitrestable(fitres)
    print(
        "Abs Diff: ",
        (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7]).mean(),
        "\t",
        (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7]).std(),
    )
    print(
        "Rel Diff: ",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).mean(),
        "\t",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).std(),
    )

if TYPE == "Test5":
    print_fitrestable(fitres)
    print(
        "Abs Diff: ",
        (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7]).mean(),
        "\t",
        (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7]).std(),
    )
    print(
        "Rel Diff: ",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).mean(),
        "\t",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).std(),
    )
    print(
        "Rel Diff to other: ",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres2)[:, 0]
        ).mean(),
        "\t",
        (
            (np.asarray(fitres)[:, 0] - np.asarray(fitres)[:, 7])
            / np.asarray(fitres2)[:, 0]
        ).std(),
    )

    print_fitrestable(fitres2)
    print(
        "Abs Diff: ",
        (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7]).mean(),
        "\t",
        (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7]).std(),
    )
    print(
        "Rel Diff: ",
        (
            (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7])
            / np.asarray(fitres2)[:, 0]
        ).mean(),
        "\t",
        (
            (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7])
            / np.asarray(fitres2)[:, 0]
        ).std(),
    )
    print(
        "Rel Diff to other: ",
        (
            (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).mean(),
        "\t",
        (
            (np.asarray(fitres2)[:, 0] - np.asarray(fitres2)[:, 7])
            / np.asarray(fitres)[:, 0]
        ).std(),
    )
