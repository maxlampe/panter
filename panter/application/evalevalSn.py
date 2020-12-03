"""For plotting and evaluation drift sn results from evalSn.py"""

import configparser
import datetime

import numpy as np
import matplotlib.pyplot as plt

import dataPerkeo as dP
import evalPerkeo as eP
import evalFitSettings as eFS


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read("evalRaw.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]

file = dP.FilePerkeo(outputFileDir + "Drift_Sn_EMG_2020-09-17.p")
# results[i] = datepair, pmt_np, gof, mu_val, mu_err
imp = file.imp()  # imp[633files][4results][16PMTs]
# FIXME: PMT15 missing?

data = imp.T

pmt_data = []
pmt_dataErr = []
PMTChi = []
x_dates = []
x_realdates = []
# Det 1 Central PMTs, Det 1 Corner PMTs, Det 2 Central PMTs, Det 2 Corner PMTs
pmt_selec = np.array([[1, 2, 5, 6], [0, 3, 4, 7], [9, 10, 13, 14], [8, 11, 12, 15]])

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Sn Drift study")

axs[0].set_title("Peak pos over time")
axs[1].set_title("rChi2 over time")
axs.flat[0].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
axs.flat[1].set(xlabel="Time [s]", ylabel="rChi2 [ ]")


for PMT in [int(input())]:
    for i, val in enumerate(imp):
        curr_err = data[4][i][PMT]
        if curr_err is not None and curr_err < 1000.0:
            x_dates.append(data[0][i][0])
            x_realdates.append(datetime.datetime.utcfromtimestamp(data[0][i][0]))
            pmt_data.append(data[3][i][PMT])
            pmt_dataErr.append(curr_err)
            PMTChi.append(data[2][i][PMT]["rChi2"])

    xdat = np.arange(1, len(pmt_data) + 1, 1)
    xdat = x_dates

    axs[0].errorbar(xdat, pmt_data, yerr=pmt_dataErr, fmt=".", label=f"PMT{PMT}")
    axs[1].plot(xdat, PMTChi, ".", label=f"PMT{PMT}")

    residuals_data = np.array(PMTChi)
    residual_hist = dP.HistPerkeo(
        residuals_data, 6, residuals_data.min() * 1.1, residuals_data.max() * 1.1
    )

    fitclass = eP.DoFit(residual_hist.hist)
    fitclass.setup(eFS.gaus_simp)
    fitclass.set_fitparam("mu", valpar=0.0)
    fitclass.set_fitparam("sig", valpar=0.0005)
    fitclass.set_fitparam("norm", valpar=0.5)
    residual_gausfit = fitclass.fit()
    axs[1].plot(
        xdat, [residual_gausfit.params["mu"].value] * len(xdat), label=f"mean PMT{PMT}"
    )

    print(residual_gausfit.fit_report())

axs[0].legend()
axs[1].legend()
plt.show()
