"""Dump"""

import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.core.dataloaderPerkeo import DLPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
hist_trigger = [None]*2

for primary_detector in [0, 1]:
    master_both = None
    master_onlysec = None
    det_prim = primary_detector
    det_sec = 1 - det_prim
    for i in range(50):
        meas = dataloader[100 + i]
        data = dP.RootPerkeo(meas.file_list[0])
        data.info()

        data.set_filtdef()
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_sec,
        )
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=4e9,
            up_lim=4e10,
            index=det_prim,
        )
        data.auto(1)
        data.gen_hist([])
        hist_onlysec = data.hist_sums

        data.set_filtdef()
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_sec,
        )
        data.set_filt(
            "data",
            fkey="CoinTime",
            active=True,
            ftype="num",
            low_lim=0,
            up_lim=4e9,
            index=det_prim,
        )
        data.set_filt(
            "data", fkey="Detector", active=True, ftype="bool", rightval=det_prim
        )
        data.auto(1)
        data.gen_hist([])
        hist_both = data.hist_sums
        hist_onlysec[det_prim].addhist(hist_both[det_prim])

        if i == 0:
            master_onlysec = hist_onlysec[det_prim]
            master_both = hist_both[det_prim]
        else:
            master_onlysec.addhist(hist_onlysec[det_prim])
            master_both.addhist(hist_both[det_prim])

    master_both.divbyhist(master_onlysec)
    hist_trigger[det_prim] = master_both
    # hist_trigger.plt([0, 12e3, 0, 1])

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
a:  0.00275708 +/- 4.0498e-05 (1.47%) (init = 0.003)
p:  0.70621967 +/- 0.00863997 (1.22%) (init = 0.655)
a:  0.00325521 +/- 7.5923e-05 (2.33%) (init = 0.003)
p:  0.61569591 +/- 0.01218149 (1.98%) (init = 0.655)
"""