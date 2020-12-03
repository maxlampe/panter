"""For fitting ElecTest Test4 and 5 data and return results into file."""

import configparser

import numpy as np
import pandas as pd

import dataPerkeo as dP
import evalPerkeo as eP
import dataFiles as dF


# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read("evalElec.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]
LOGFILE = cnf["DEFAULT"]["logfile"]

BSigFix = cnf["evalFits"].getboolean("BSigFix")
BNorFix = cnf["evalFits"].getboolean("BNorFix")
BFitAll = cnf["evalFits"].getboolean("BFitAll")

filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_electest(
    mtype="Test5",
    fit_mode=0,
    fanout_mode=input(),
    bfitall=BFitAll,
    bfixsig=BSigFix,
    bfixnorm=BNorFix,
)

delt, minIterFilter, maxIterFilter, IterforSigmaMean = [
    float(cnf["evalFits"]["Eval_DPTT_Delta"]),
    int(cnf["evalFits"]["Eval_DPTT_imin"]),
    int(cnf["evalFits"]["Eval_DPTT_iMax"]),
    int(cnf["evalFits"]["Eval_DPTT_iSig"]),
]

ITER = 0
for lis in filelist:
    for j, j_lis in enumerate(lis):
        # currentFile
        currRes = []
        currResFixed = []
        if bList[1]:
            data = dP.RootPerkeo(j_lis)
            data.info()
            # do once with all fit params free for a lim. range for high DPTT
            for i in range(1, 2):
                data.set_filtdef()
                data.datafilter["Cycle"].active = True
                data.datafilter["DeltaPrevTriggerTime"].active = True
                data.datafilter["DeltaPrevTriggerTime"].upperlimit = (
                    delt * maxIterFilter + delt * 0.5
                )
                data.datafilter["DeltaPrevTriggerTime"].lowerlimit = (
                    delt * (maxIterFilter - IterforSigmaMean) - delt * 0.5
                )
                data.auto(1)
                print("Active PMTs:\t", data.ret_actpmt())
                data.gen_hist(data.ret_actpmt())

                fitdataclass = eP.DoFitData(data, "ElecTest5")
                fit_result = fitdataclass.fit()

                currRes.append(fit_result)

            del data

            listSigmas2D = []
            listSigmas0 = list(
                map(
                    lambda x: x.params["sig"].value,
                    (np.array(currRes).T[3]).flatten(),
                )
            )
            listSigmas1 = list(
                map(
                    lambda x: x.params["sig"].value,
                    (np.array(currRes).T[4]).flatten(),
                )
            )
            listSigmas2D = np.zeros([len(listSigmas0), 2])
            for i, val in enumerate(listSigmas2D):
                listSigmas2D[i, 0] = abs(listSigmas0[i])
                listSigmas2D[i, 1] = abs(listSigmas1[i])

            print("Starting round two for fixed sigmas.")

        data = dP.RootPerkeo(j_lis)
        # DelPTT[MYCROSECONDS!]
        dptt_values = (
            0.01
            * delt
            * np.linspace(
                minIterFilter, maxIterFilter - 1, maxIterFilter - minIterFilter
            )
        )

        for i in range(minIterFilter, maxIterFilter):
            currFilt = delt * i
            print(
                "\n-------------"
                f"{100.*(float(i)/maxIterFilter):.2f} %%"
                f"( {currFilt:.1f} )-------------\n"
            )

            data.set_filtdef()
            data.datafilter["Cycle"].active = True
            data.datafilter["DeltaPrevTriggerTime"].active = True
            data.datafilter["DeltaPrevTriggerTime"].upperlimit = delt * i + delt * 0.5
            data.datafilter["DeltaPrevTriggerTime"].lowerlimit = delt * i - delt * 0.5
            data.auto(1)

            print("Active PMTs:\t", data.ret_actpmt())
            if bList[0]:
                data.gen_hist(list(range(0, data.no_pmts)))
            else:
                data.gen_hist(data.ret_actpmt())

            if bList[1]:
                for jj, pmt_ind in enumerate(data.ret_actpmt()):
                    data.stats["sig2D"][pmt_ind] = listSigmas2D[jj]

            fitdataclass = eP.DoFitData(data, "ElecTest5_fixSig")
            fit_result = pd.DataFrame(
                fitdataclass.fit(),
                columns=(["PMT", "GoF1", "GoF2", "ModelRes1", "ModelRes2"]),
            )
            currResFixed.append(fit_result)
        del data
        s_onefile = pd.Series(currResFixed, index=dptt_values)
        results[ITER][j] = s_onefile
    ITER += 1

print(results)

file = dP.FilePerkeo(outputFileDir + EXPFILE)
print("Write to file", file.dump(results))
# write settings to Log file
fitdataclass.write_2log(outputFileDir + LOGFILE)

print("\n \t\tDing Dong Done. \n")
