"""Dump"""

import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
meas = dataloader[100]
data = dP.RootPerkeo(meas.file_list[0])
data.auto()
data.gen_hist([])

print(data.hist_sums[0].hist)

data.hist_sums[0].write2root("DetSum0", "inter")
data.hist_sums[1].write2root("DetSum1", "inter", True)
data.hist_sums[1].plt()
