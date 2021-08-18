""""""

import sys
import numpy as np
import matplotlib.pyplot as plt

from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrSimple import CorrSimple
from panter.core.dataPerkeo import RootPerkeo, HistPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 69536])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])

data_class = RootPerkeo(filt_meas[0].file_list[0])
data_class.set_filtdef()
data_class.set_filt(
    "data",
    fkey="CoinTime",
    active=True,
    ftype="num",
    low_lim=0,
    up_lim=2e9,
    index=0,
)
data_class.set_filt(
    "data",
    fkey="CoinTime",
    active=True,
    ftype="num",
    low_lim=0,
    up_lim=2e9,
    index=1,
)
data_class.auto(1)

coin_arr = data_class.ret_array_by_key(key="CoinTime").transpose()
coin0 = np.array(coin_arr[0], dtype=float)
coin1 = np.array(coin_arr[1], dtype=float)
diff = coin0 - coin1

plt.hist(diff, range=[-30, 30])
plt.show()
hist = HistPerkeo(diff, **{"bin_count": 200, "low_lim": -20, "up_lim": 20})
hist.plot_hist(title="Raw E-ToF")

corr_class = CorrSimple(
    dataloader=filt_meas,
    branch_key="CoinTime",
    hist_par={"bin_count": 200, "low_lim": -20, "up_lim": 20},
)
corr_class.addition_filters.append(
    {
        "tree": "data",
        "fkey": "CoinTime",
        "active": True,
        "ftype": "num",
        "low_lim": 0,
        "up_lim": 2000000000,
        "index": 0,
    }
)
corr_class.addition_filters.append(
    {
        "tree": "data",
        "fkey": "CoinTime",
        "active": True,
        "ftype": "num",
        "low_lim": 0,
        "up_lim": 2000000000,
        "index": 1,
    }
)

corr_class.corr(bstore=True, bwrite=False)
for ind, hist in corr_class.histograms:
    for det in range(len(hist)):
        hist[det].plot_hist(title="E-ToF - corrected")
