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
            mtype="Test5",
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
            [9.401640881673009],  # 0.2076543018779534
            [50.40699636466046],  # 0.8711523782767728
            [21.917254461004834],  # 0.8830771320349838
            [-26.588244716096614],  # 0.37922626094907697
            [33.07727524327549],  # 0.4153803165154893
            [24.053441315222756],  # 1.0271641428509521
            [8.54766787369338],  # 0.4779033771529284
            [-16.357538315216754],  # 0.5827911535095025
            [-18.22768281584457],  # 0.6320068178633943
            [24.118855334539063],  # 0.6864848540696089
            [16.194634025569865],  # 1.1767831194769591
            [131.08845406625397],  # 0.6924975848860111
            [54.02472643807953],  # 0.5964033207663155
            [41.417907479294826],  # 0.5568582864951819
            [15.650696054486128],  # 0.47892160264545036
            [22.68588745139067],  # 0.40900195829860986
        ]
    )


for fan in [0, 1, 2, 3]:
    FANOUT = fan
    PMTS = list(range(FANOUT * 4, FANOUT * 4 + 4))

    filelist, results, FITMOD, EXPFILE, bList = dF.setupeval_electest(
        mtype="Test5",
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

                # do fit (single gaussian)
                listSigmas2D = [[None, None]] * 16
                for pmt, hist in enumerate(corr_class.histograms[0][1]):
                    if hist.hist is not None and pmt in PMTS:
                        # print(hist.stats["mean"])
                        # print(hist.stats["std"])
                        # hist.plot_hist()
                        for ampl in [0, 1]:
                            dofitclass = eP.DoFit(hist.hist)
                            dofitclass.setup(eFS.gaus_gen)
                            # dofitclass.set_bool("boutput", True)
                            dofitclass.set_bool("boutput", False)
                            dofitclass.set_fitparam(
                                namekey="mu",
                                bparafree=True,
                                valpar=(
                                    hist.stats["mean"]
                                    + (-1 + 2 * ampl) * hist.stats["std"]
                                ),
                            )
                            fit_result = dofitclass.fit()

                            currRes.append(fit_result)
                            if fit_result is not None:
                                listSigmas2D[pmt][ampl] = fit_result.params["sig"].value

                print(listSigmas2D)

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
                    f"{100.*(float(i)/maxIterFilter):.2f} %"
                    f"( {currFilt:.1f} )-------------\n"
                )

                # reset corrPerkeo filter and add new filter

                # maybe fill dataloader with all events instead of picking last?

                # FIXME: Why do I need to reinstantiate instead of corr_class.clear()?
                events = [[2, 8, [j_lis]]]
                dataloader = DLPerkeo("")
                dataloader.fill(events)
                # maybe fill dataloader with all events instead of picking last?
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
                fit_result_df = pd.DataFrame(
                    columns=["PMT", "GoF1", "GoF2", "ModelRes1", "ModelRes2"]
                )
                for pmt, hist in enumerate(corr_class.histograms[0][1]):
                    if hist.hist is not None and pmt in PMTS:
                        inter_res = []
                        for ampl in [0, 1]:
                            dofitclass = eP.DoFit(hist.hist)
                            dofitclass.setup(eFS.gaus_gen)
                            # dofitclass.set_bool("boutput", True)
                            dofitclass.set_bool("boutput", False)
                            dofitclass.set_fitparam(
                                namekey="mu",
                                bparafree=True,
                                valpar=(
                                    hist.stats["mean"]
                                    + (-1 + 2 * ampl) * hist.stats["std"]
                                ),
                            )
                            if bList[1]:
                                dofitclass.set_fitparam(
                                    namekey="sig",
                                    bparafree=False,
                                    valpar=listSigmas2D[pmt][ampl],
                                )
                            fit_result = dofitclass.fit()
                            inter_res.append(dofitclass.ret_gof())
                            inter_res.append(fit_result)

                        # FIXME: Need to pass results
                        res_dict = {
                            "PMT": pmt,
                            "GoF1": inter_res[0],
                            "GoF2": inter_res[2],
                            "ModelRes1": inter_res[1],
                            "ModelRes2": inter_res[3],
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
