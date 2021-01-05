"""Dump"""

import numpy as np
import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

meas = dataloader[100]

data = dP.RootPerkeo(meas.file_list[0])
data.info()
data.print_filt()

data.set_filtdef()
data.set_filt(
    "data", fkey="CoinTime", active=True, ftype="num", low_lim=0, up_lim=4e9, index=0
)
data.print_filt()
data.set_filt(
    "data", fkey="CoinTime", active=True, ftype="num", low_lim=0, up_lim=4e9, index=1
)
data.print_filt()
data.auto(1)
