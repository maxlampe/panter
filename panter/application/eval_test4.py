"""For fitting ElecTest Test4 and 5 data and return results into file."""

import configparser

import numpy as np
import pandas as pd

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.core.dataFiles as dF

from panter.config import conf_path

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalElec.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]
LOGFILE = cnf["DEFAULT"]["logfile"]

BSigFix = cnf["evalFits"].getboolean("BSigFix")
BNorFix = cnf["evalFits"].getboolean("BNorFix")
BFitAll = cnf["evalFits"].getboolean("BFitAll")

filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_electest(
    mtype="Test4",
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
                data.clear_filt()
                data.set_filt(
                    "data",
                    fkey="DeltaPrevTriggerTime",
                    active=True,
                    ftype="num",
                    low_lim=delt * (maxIterFilter - IterforSigmaMean) - delt * 0.5,
                    up_lim=delt * maxIterFilter + delt * 0.5,
                )
                data.auto(1)

                print("Active PMTs:\t", data.ret_actpmt())
                data.gen_hist(data.ret_actpmt())

                fitdataclass = eP.DoFitData(data, "ElecTest4")
                fit_result = fitdataclass.fit()

                currRes.append(fit_result)

            del data

            sigmares = list(
                map(
                    lambda x: x.params["sig"].value,
                    (np.array(currRes).T[2]).flatten(),
                )
            )
            listSigmas = []
            for i in sigmares:
                listSigmas.append(i.mean())

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
            data.set_filt(
                "data",
                fkey="DeltaPrevTriggerTime",
                active=True,
                ftype="num",
                low_lim=delt * i - delt * 0.5,
                up_lim=delt * i + delt * 0.5,
            )
            data.auto(1)

            print("Active PMTs:\t", data.ret_actpmt())
            if bList[0]:
                data.gen_hist(list(range(0, data.no_pmts)))
            else:
                data.gen_hist(data.ret_actpmt())

            if bList[1]:
                # set internal sigma values to expected overall mean
                for jj, pmt_ind in enumerate(data.ret_actpmt()):
                    data.stats["sig"][pmt_ind] = listSigmas[jj]

            fitdataclass = eP.DoFitData(data, "ElecTest4_fixSig")
            fit_result = pd.DataFrame(
                fitdataclass.fit(), columns=(["PMT", "GoF", "ModelRes"])
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
