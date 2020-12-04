"""Dump"""

import numpy as np
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo


import pandas as pd

bincent = [0, 1, 2, 3, 4, 5]
hist = [0, 1, 3, 2, 1, 0]
hist0 = pd.DataFrame({"x": bincent, "y": hist, "err": np.sqrt(hist)})


def filt0(histDF: pd.DataFrame) -> pd.DataFrame:
    """"""

    filt = histDF["y"] >= 1
    histDF = histDF[filt]

    return histDF

print(hist0)
print(filt0(hist0))

exit()

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


corr_class = corrPerkeo(dataloader[0:1])
corr_class.corrections["Pedestal"] = True
corr_class.corrections["RateDepElec"] = True
corr_class.corr()
