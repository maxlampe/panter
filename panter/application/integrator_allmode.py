"""Studying raw integrator values over time from ALLMODE measurements"""

import numpy as np
import matplotlib.pyplot as plt
from panter.core.dataPerkeo import RootPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dir3 = "/mnt/sda/PerkeoDaten1920/cycle201/Det200123/"
filenames_both = ["data201932.root", "data201936.root"]
filenames_delta = ["data201948.root", "data201952.root"]
filenames_all = ["data201964-70606_bg.root", "data201980-70606_3.root"]

pmt = 1

fig, axs = plt.subplots(1, 2, figsize=(12, 7))
fig.suptitle("ALLMODE events Integrator values")
for ind in range(2):
    data_class = RootPerkeo(dir + filenames_all[1])
    data_class.info()
    data_class.auto()

    pmt_all = np.transpose(data_class.ret_array_by_key("PMT"), (1, 0, 2))
    pmt0 = pmt_all[pmt + ind * 8]
    axs.flat[ind].set_title(f"PMT{pmt + ind * 8}")
    axs.flat[ind].set(xlabel="TDC [10ns]", ylabel="ADC [ch]")

    for event in range(15):
        axs.flat[ind].plot(pmt0[event])

plt.savefig("../output/integrator_allmode.png", dpi=300)
plt.show()
