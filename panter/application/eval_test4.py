"""For fitting ElecTest Test4 and 5 data and return results into file."""

import configparser

import numpy as np
import pandas as pd

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.core.dataFiles as dF
import panter.config.evalFitSettings as eFS

from panter.config import conf_path
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalElec.ini")

outputFileDir = cnf["DEFAULT"]["expDir"]
LOGFILE = cnf["DEFAULT"]["logfile"]

BSigFix = cnf["evalFits"].getboolean("BSigFix")
BNorFix = cnf["evalFits"].getboolean("BNorFix")
BFitAll = cnf["evalFits"].getboolean("BFitAll")

delt, minIterFilter, maxIterFilter, IterforSigmaMean = [
    float(cnf["evalFits"]["Eval_DPTT_Delta"]),
    int(cnf["evalFits"]["Eval_DPTT_imin"]),
    int(cnf["evalFits"]["Eval_DPTT_iMax"]),
    int(cnf["evalFits"]["Eval_DPTT_iSig"]),
]


# do pedestal eval for all files.
if 0:
    PEDESTALS = [[] for i in range(16)]

    for fan in [0, 1, 2, 3]:
        FANOUT = fan
        # FANOUT = input()
        PMTS = list(range(FANOUT * 4, FANOUT * 4 + 4))

        filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_electest(
            mtype="Test4",
            fit_mode=0,
            fanout_mode=FANOUT,
            bfitall=BFitAll,
            bfixsig=BSigFix,
            bfixnorm=BNorFix,
        )

        dataloader = DLPerkeo("")
        for ind_lis, lis in enumerate(filelist):
            for j, j_lis in enumerate(lis):
                # currentFile
                currRes = []
                currResFixed = []

                data = dP.RootPerkeo(j_lis)
                pedtest = eP.PedPerkeo(
                    dataclass=data,
                    bplot_res=False,
                    bplot_fit=False,
                    bplot_log=True,
                    bnaive_filt=True,
                    bfilt_detsum=False,
                )
                res = pedtest.ret_pedestals()
                for pmt, ped in enumerate(res):
                    if pmt not in PMTS:
                        print(f"{pmt}\t{ped[0]:0.3f}\t{ped[1]:0.3f}")
                        PEDESTALS[pmt].append(ped[0])
                    else:
                        print(f"{pmt}\t{ped}")
                # print(pedtest.ret_pedestals().T[0].sum())
                # print(np.sqrt((pedtest.ret_pedestals().T[1] ** 2).sum()))
                events = [[2, 8, [j_lis]]]
                dataloader.fill(events)

    print(PEDESTALS)
    for ind, pmt_ped in enumerate(PEDESTALS):
        pmt_ped = np.asarray(pmt_ped)
        print(f"PMT{ind}\t{pmt_ped.mean()}\t{pmt_ped.std()}")

    exit()

else:
    ped_avg = np.asarray(
        [
            [9.586288111952882],  # 0.3771269915326741
            [49.79487607252694],  # 0.7161614352448196
            [21.92457573325795],  # 0.528157434483814
            [-27.382442864804567],  # 0.917731792943326
            [34.33428061488849],  # 2.5713806621438517
            [24.422737039537207],  # 2.0898846997041662
            [9.062653897149024],  # 0.43836235397986817
            [-16.4859432461382],  # 0.6570451632768408
            [-19.226079233823853],  # 0.45936434247538577
            [24.855495862342107],  # 1.8554495752252915
            [17.09170930965023],  # 2.283215547844795
            [131.9052882092843],  # 1.9605564890392135
            [54.82903995127997],  # 5.757309766664769
            [42.02061749298743],  # 5.035326781862424
            [14.304248412104862],  # 1.793381954649485
            [21.91363310100641],  # 3.3202969367850903
        ]
    )

for fan in [0, 1, 2, 3]:
    FANOUT = fan
    PMTS = list(range(FANOUT * 4, FANOUT * 4 + 4))

    filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_electest(
        mtype="Test4",
        fit_mode=0,
        fanout_mode=FANOUT,
        bfitall=BFitAll,
        bfixsig=BSigFix,
        bfixnorm=BNorFix,
    )

    for ind_lis, lis in enumerate(filelist):
        for j, j_lis in enumerate(lis):
            # if j > 0:
            #    continue
            # currentFile
            currRes = []
            currResFixed = []

            events = [[2, 8, [j_lis]]]
            dataloader = DLPerkeo("")
            dataloader.fill(events)
            # maybe fill dataloader with all events instead of picking last?
            meas = dataloader.ret_meas()[-1]
            corr_class = CorrPerkeo(meas, mode=2, ped_arr=ped_avg)
            corr_class.set_all_corr(bactive=False)
            corr_class.corrections["Pedestal"] = True

            if bList[1]:
                # if bfixsig
                # filter corrPerkeo for large, high DPTT range to get sigma value
                corr_class.addition_filters.append(
                    {
                        "tree": "data",
                        "fkey": "DeltaPrevTriggerTime",
                        "active": True,
                        "ftype": "num",
                        "low_lim": delt * (maxIterFilter - IterforSigmaMean)
                        - delt * 0.5,
                        "up_lim": delt * maxIterFilter + delt * 0.5,
                    }
                )
                corr_class.corr(bstore=True, bwrite=False)

                """
                # only active PMTs?
                print("Active PMTs:\t", data.ret_actpmt())
                data.gen_hist(data.ret_actpmt())
                """

                # do fit (single gaussian)
                listSigmas = [None] * 16
                for pmt, hist in enumerate(corr_class.histograms[0][1]):
                    if hist.hist is not None and pmt in PMTS:
                        # hist.plot_hist()
                        dofitclass = eP.DoFit(hist.hist)
                        dofitclass.setup(eFS.gaus_gen)
                        dofitclass.set_bool("boutput", False)
                        fit_result = dofitclass.fit()

                        currRes.append(fit_result)
                        if fit_result is not None:
                            listSigmas[pmt] = fit_result.params["sig"].value

                print(listSigmas)
                """
                # sigmares = [x.params["sig"].value for x in currRes]
                sigmares = [
                    x.params["sig"].value if x is not None else None for x in currRes
                ]

                listSigmas = []
                for i in sigmares:
                    listSigmas.append(i.mean())
                """
                print("Starting round two for fixed sigmas.")

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
                    f"{100.*(float(i)/maxIterFilter):.2f} \%"
                    f"( {currFilt:.1f} )-------------\n"
                )

                # reset corrPerkeo filter and add new filter

                # maybe fill dataloader with all events instead of picking last?

                # FIXME: Why do I need to reinstantiate instead of corr_class.clear()?
                events = [[2, 8, [j_lis]]]
                dataloader.fill(events)
                meas = dataloader.ret_meas()[-1]
                corr_class = CorrPerkeo(meas, mode=2, ped_arr=ped_avg)
                corr_class.set_all_corr(bactive=False)
                corr_class.corrections["Pedestal"] = True

                corr_class.addition_filters.append(
                    {
                        "tree": "data",
                        "fkey": "DeltaPrevTriggerTime",
                        "active": True,
                        "ftype": "num",
                        "low_lim": delt * i - delt * 0.5,
                        "up_lim": delt * i + delt * 0.5,
                    }
                )
                corr_class.corr(bstore=True, bwrite=False)

                # do fit (single gaussian)
                fit_result_df = pd.DataFrame(columns=["PMT", "GoF", "ModelRes"])
                for pmt, hist in enumerate(corr_class.histograms[0][1]):
                    if hist.hist is not None and pmt in PMTS:
                        dofitclass = eP.DoFit(hist.hist)
                        dofitclass.setup(eFS.gaus_gen)
                        dofitclass.set_bool("boutput", False)
                        if bList[1]:
                            dofitclass.set_fitparam(
                                namekey="sig", bparafree=False, valpar=listSigmas[pmt]
                            )
                        dofitclass.set_fitparam(namekey="mu", valpar=hist.stats["mean"])
                        fit_result = dofitclass.fit()

                        res_dict = {
                            "PMT": pmt,
                            "GoF": dofitclass.ret_gof(),
                            "ModelRes": fit_result,
                        }

                        fit_result_df = fit_result_df.append(
                            res_dict, ignore_index=True
                        )

                currResFixed.append(fit_result_df)

                """
                # old fit result dim:
                # [i, dofitclass.ret_gof(), dofitclass.ret_results()]
                # gof probably still dict
                # i = PMT no
    
                dofitclass = DoFit(self._dataclass.hists[i].hist)
                dofitclass.setup(self._fitsettings)
                dofitclass.set_bool("boutput", self.boutput)
                # if fix sigma
                if self._datatype == self._valid_datatypes[3]:
                    dofitclass.set_fitparam(namekey="sig", bparafree=False)
                dofitclass.fit()
                """

                """
                fitdataclass = eP.DoFitData(data, "ElecTest4_fixSig")
                fit_result = pd.DataFrame(
                    fitdataclass.fit(), columns=(["PMT", "GoF", "ModelRes"])
                )
                currResFixed.append(fit_result)
                """
            s_onefile = pd.Series(currResFixed, index=dptt_values)
            results[ind_lis][j] = s_onefile

    print(results)

    file = dP.FilePerkeo(EXPFILE)
    print("Write to file", file.dump(results))

print("\n \t\tDing Dong Done. \n")
