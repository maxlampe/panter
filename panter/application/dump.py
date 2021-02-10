"""Dump"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 3, 67732])

meas = filt_meas[0:50]

corr_class = corrPerkeo(dataloader=meas, mode=0)
corr_class.corrections["Pedestal"] = True
corr_class.corrections["RateDepElec"] = True
corr_class.corrections["DeadTime"] = True
corr_class.corrections["Drift"] = False

corr_class.addition_filters.append(
    {
        "tree": "data",
        "fkey": "Detector",
        "active": True,
        "ftype": "bool",
        "rightval": 0,
    }
)

corr_class.corr(bstore=True, bwrite=False)


hists = np.array([])
for entry in corr_class.histograms:
    hists = np.append(hists, entry[1][0])

final = dP.average_hists(hists)
corr_class.histograms[0][1][0].plt()
final.plt()
