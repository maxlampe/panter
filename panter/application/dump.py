"""Dump"""

import numpy as np
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

# TODO: make dataloader.info()
length = dataloader.length()
print("Data loader length\n", length)


print(dataloader[1400])
print(dataloader[1400]())

print(dataloader[-4:-1])
dataloader.print([5, 12])

meas = dataloader.ret_filt_meas(["tp", "src"], [1, 2])
print(type(meas))
print(meas)


exit()

corr_class = corrPerkeo(dataloader[0:4])
corr_class.corr()

