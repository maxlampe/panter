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
# filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 3])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 3, 67732])

meas = dataloader[0]

corr_class = corrPerkeo(dataloader=meas, mode=2)
corr_class.corrections["Pedestal"] = False
corr_class.corrections["RateDepElec"] = False
corr_class.corrections["DeadTime"] = True
corr_class.corr(bstore=True, bwrite=False)

exit()

fitsettings = eFS.gaus_expmod
fitsettings.plot_labels = [
    "SnSpec fit result",
    "ADC [ch]",
    "Counts [ ]",
]
fitsettings.plotrange["x"] = [30.0, 4000.0]

hists = corr_class.histograms[0][1]
i = 0

dofitclass = eP.DoFit(hists[i].hist)
dofitclass.setup(fitsettings)
dofitclass.set_bool("boutput", True)
dofitclass.fit()

print([i, dofitclass.ret_gof(), dofitclass.ret_results()])
