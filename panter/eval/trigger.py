"""Trigger analysis"""

from panter.config.evalFitSettings import trigger_func
from panter.data.dataMeasPerkeo import MeasPerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.corrPerkeo import CorrPerkeo
from panter.eval.evalFit import DoFit

dir_path = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir_path)
dataloader.auto()
hist_trigger = [None] * 2


def trigger_raw(meas: MeasPerkeo, det_main: int):
    """Calculate trigger function for one detector from raw data."""

    det_bac = 1 - det_main
    data = RootPerkeo(meas.file_list[0])
    data.info()

    data.set_filtdef()
    data.set_filt(
        "data",
        fkey="CoinTime",
        active=True,
        ftype="num",
        low_lim=0,
        up_lim=4e9,
        index=det_bac,
    )
    data.set_filt(
        "data",
        fkey="CoinTime",
        active=True,
        ftype="num",
        low_lim=4e9,
        up_lim=4e10,
        index=det_main,
    )
    data.auto(1)
    data.gen_hist([])
    hist_onlybac = data.hist_sums

    data.set_filtdef()
    data.set_filt(
        "data",
        fkey="CoinTime",
        active=True,
        ftype="num",
        low_lim=0,
        up_lim=4e9,
        index=det_bac,
    )
    data.set_filt(
        "data",
        fkey="CoinTime",
        active=True,
        ftype="num",
        low_lim=0,
        up_lim=4e9,
        index=det_main,
    )
    data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=det_main)
    data.auto(1)
    data.gen_hist([])

    hist_b = data.hist_sums
    hist_onlybac[det_main].addhist(hist_b[det_main])

    return [hist_b, hist_onlybac]


def trigger_corr(meas: MeasPerkeo, det_main: int):
    """Calculate trigger function for one detector from corrected data."""

    det_bac = 1 - det_main
    corr_class = CorrPerkeo(dataloader=meas, mode=1)
    corr_class.set_all_corr(bactive=False)
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
            "index": det_bac,
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
            "index": det_main,
        }
    )
    corr_class.corr(bstore=True, bwrite=False)
    hist_onlybac = corr_class.histograms[0][0]

    corr_class.clear()

    corr_class.addition_filters.append(
        {
            "tree": "data",
            "fkey": "CoinTime",
            "active": True,
            "ftype": "num",
            "low_lim": 0,
            "up_lim": 4e9,
            "index": det_bac,
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
            "index": det_main,
        }
    )
    corr_class.addition_filters.append(
        {
            "tree": "data",
            "fkey": "Detector",
            "active": True,
            "ftype": "bool",
            "rightval": det_main,
        }
    )
    corr_class.corr(bstore=True, bwrite=False)

    hist_b = corr_class.histograms[0][0]
    hist_onlybac[det_main].addhist(hist_b[det_main])

    return [hist_b, hist_onlybac]


for primary_detector in [0, 1]:
    master_both = None
    master_onlysec = None
    det_prim = primary_detector
    det_sec = 1 - det_prim
    for i in range(5):
        meas = dataloader[100 + i]

        if False:
            hist_both, hist_onlysec = trigger_corr(meas, det_prim)
        else:
            hist_both, hist_onlysec = trigger_raw(meas, det_prim)

        if i == 0:
            master_onlysec = hist_onlysec[det_prim]
            master_both = hist_both[det_prim]
        else:
            master_onlysec.addhist(hist_onlysec[det_prim])
            master_both.addhist(hist_both[det_prim])

    master_both.divbyhist(master_onlysec)
    hist_trigger[det_prim] = master_both

for hist in hist_trigger:
    fitclass = DoFit(hist.hist)
    fitclass.setup(trigger_func)
    fitclass.limit_range([500, 15e3])
    fitclass.set_fitparam(namekey="a", valpar=0.003)
    fitclass.set_fitparam(namekey="p", valpar=0.655)
    fitclass.set_bool("boutput", True)
    fitclass.plotrange["x"] = [0, 15e3]
    fitclass.plotrange["y"] = [-0.2, 1.2]
    fitclass.plot_labels = ["Trigger Det", "ADC [ch]", "Trigger prob. [ ]"]
    fitclass.fit()

"""
Raw:

a:  0.00275708 +/- 4.0498e-05 (1.47%) (init = 0.003)
p:  0.70621967 +/- 0.00863997 (1.22%) (init = 0.655)
a:  0.00325521 +/- 7.5923e-05 (2.33%) (init = 0.003)
p:  0.61569591 +/- 0.01218149 (1.98%) (init = 0.655)

bgsub, nocorr:
a:  0.00253116 +/- 4.5523e-05 (1.80%) (init = 0.003)
p:  0.73623919 +/- 0.01119634 (1.52%) (init = 0.655)
a:  0.00287972 +/- 7.5364e-05 (2.62%) (init = 0.003)
p:  0.68840456 +/- 0.01481232 (2.15%) (init = 0.655)

bgsub, corr:
a:  0.00251647 +/- 3.1369e-05 (1.25%) (init = 0.003)
p:  0.73960438 +/- 0.00779516 (1.05%) (init = 0.655)
a:  0.00287115 +/- 5.2395e-05 (1.82%) (init = 0.003)
p:  0.68867295 +/- 0.01034353 (1.50%) (init = 0.655)
"""
