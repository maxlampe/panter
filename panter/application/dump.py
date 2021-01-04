"""Dump"""

import numpy as np
import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

# TODO: make dataloader.info()
length = dataloader.length()
print("Data loader length\n", length)

corr_class = corrPerkeo(dataloader[0])
corr_class.corrections["Pedestal"] = False
corr_class.corrections["RateDepElec"] = False
corr_class.corr()
