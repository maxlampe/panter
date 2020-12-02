"""For fitting drift sn data and return results into file."""

import configparser

import numpy as np

import dataPerkeo as dP
import evalPerkeo as eP
import dataFiles as dF


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read("evalDrift.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]
LOGFILE = cnf["DEFAULT"]["logfile"]

BFitAll = cnf["evalFits"].getboolean("BFitAll")

filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_driftmeas(
    mtype="Drift", fit_mode=5, bfitall=False, bpair=True
)

for i, pair in enumerate(filelist):
    print(i, pair)

    dataSrc, dataBg = [dP.RootPerkeo(pair[0]), dP.RootPerkeo(pair[1])]
    dataSrc.auto()
    dataBg.auto()
    dataSrc.gen_hist(list(range(0, dataSrc.no_pmts)))
    dataBg.gen_hist(list(range(0, dataBg.no_pmts)))

    for j, ival in enumerate(dataSrc.hists):
        ival.addhist(dataBg.hists[j], -dataSrc.cy_valid_no / dataBg.cy_valid_no)

    fitdataclass = eP.DoFitData(dataSrc, "DriftSn")
    fit_result = fitdataclass.fit()

    datepair = np.array([dataSrc.filedate, dataBg.filedate])
    pmt_np = np.array(list(map(lambda x: x[0], fit_result)))
    gof = np.array(list(map(lambda x: x[1], fit_result)))
    mu_val = np.array(list(map(lambda x: x[2].params["mu"].value, fit_result)))
    mu_err = np.array(list(map(lambda x: x[2].params["mu"].stderr, fit_result)))
    del dataSrc
    del dataBg
    results[i] = np.array([datepair, pmt_np, gof, mu_val, mu_err])

results = np.array(results)

file = dP.FilePerkeo(outputFileDir + EXPFILE)
print("Write to file", file.dump(results))
# write settings to Log file
eP.write2Log(outputFileDir + LOGFILE, "Drift")

print("\n \t\tDing Dong Done. \n")
