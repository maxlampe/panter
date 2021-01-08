"""Dump"""

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.core.corrPerkeo import corrPerkeo
from panter.core.dataloaderPerkeo import DLPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
hist_trigger = [None] * 2

for primary_detector in [0, 1]:
    master_both = None
    master_onlysec = None
    det_prim = primary_detector
    det_sec = 1 - det_prim

    for i in range(50):
        meas = dataloader[100 + i]

        corr_class = corrPerkeo(meas)
        corr_class.corrections["Pedestal"] = True
        corr_class.corrections["RateDepElec"] = True
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 0,
                "up_lim": 4e9,
                "index": det_sec,
            }
        )
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 4e9,
                "up_lim": 4e10,
                "index": det_prim,
            }
        )
        corr_class.corr(bstore=True, bwrite=False)
        hist_onlysec = corr_class.histograms[0][0]

        # TODO: make clear function for reusability
        corr_class.addition_filters = []
        corr_class.histograms = []
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 0,
                "up_lim": 4e9,
                "index": det_sec,
            }
        )
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "CoinTime",
                "active": True,
                "ftype": "num",
                "low_lim": 0,
                "up_lim": 4e9,
                "index": det_prim,
            }
        )
        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "Detector",
                "active": True,
                "ftype": "bool",
                "rightval": det_prim,
            }
        )
        corr_class.corr(bstore=True, bwrite=False)
        hist_both = corr_class.histograms[0][0]

        hist_onlysec[det_prim].addhist(hist_both[det_prim])

        if i == 0:
            master_onlysec = hist_onlysec[det_prim]
            master_both = hist_both[det_prim]
        else:
            master_onlysec.addhist(hist_onlysec[det_prim])
            master_both.addhist(hist_both[det_prim])

    master_both.divbyhist(master_onlysec)
    hist_trigger[det_prim] = master_both
    # hist_trigger[det_prim].plt([0, 12e3, 0, 1])

for hist in hist_trigger:
    fitclass = eP.DoFit(hist.hist)
    fitclass.setup(eFS.trigger_func)
    fitclass.limitrange([500, 15e3])
    fitclass.set_fitparam(namekey="a", valpar=0.003)
    fitclass.set_fitparam(namekey="p", valpar=0.655)
    fitclass.set_bool("boutput", True)
    fitclass.plotrange["x"] = [0, 15e3]
    fitclass.plotrange["y"] = [-0.2, 1.2]
    fitclass.plot_labels = ["Trigger Det", "ADC [ch]", "Trigger prob. [ ]"]
    fitclass.fit()

"""
nocorr:
a:  0.00253116 +/- 4.5523e-05 (1.80%) (init = 0.003)
p:  0.73623919 +/- 0.01119634 (1.52%) (init = 0.655)
a:  0.00287972 +/- 7.5364e-05 (2.62%) (init = 0.003)
p:  0.68840456 +/- 0.01481232 (2.15%) (init = 0.655)

corr:

"""
