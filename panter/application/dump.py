"""Dump"""

import numpy as np
import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo




dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
meas = dataloader[600]

data = dP.RootPerkeo(meas.file_list[0])
data.info()
data.auto()


# TODO: make dataloader.info()
