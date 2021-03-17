"""Calculate pedestals for given files"""

import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
filename2 = "data119886-67506_bg.root"
filename3 = "data119874-67506_2.root"
filename4 = "data120576-67534_bg.root"
filename5 = "data120564-67534_3.root"
filename6 = "data119754-67502_beam.root"

dir2 = (
    "/mnt/sda/PerkeoDaten1920/ElecTest_20200309/Test5_Using_double_sweep_mode_diff_Ampl"
)
filename7 = "data249733.root"
filename8 = "data244813.root"

file = dir + filename6
data = dP.RootPerkeo(file)

results = []
detsum_range = range(2000, 45000, 1000)
for detsum in detsum_range:
    pedtest = eP.PedPerkeo(
        dataclass=data,
        bplot_res=False,
        bplot_fit=False,
        bplot_log=True,
        bfilt_detsum=True,
        range_detsum=[0, detsum],
    )
    results.append(pedtest.ret_pedestals().T)

results = np.asarray(results).T

fig, axs = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle("Pedestal position for different DetSum max cuts")
fig.subplots_adjust(hspace=0.6)
fig.subplots_adjust(wspace=0.6)

for pmd_ind, pmt_val in enumerate(results):
    axs.flat[pmd_ind].errorbar(detsum_range, pmt_val[0], yerr=pmt_val[1], fmt=".")
    axs.flat[pmd_ind].set_title(f"PMT{pmd_ind}")
    axs.flat[pmd_ind].set(xlabel="DetSum cut [ch]", ylabel="Ped [ch]")

plt.tight_layout()
plt.savefig("../output/pedestal_detsum_maxcuts.png")
plt.show()
