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
detsum_range = range(4000, 40000, 500)
for detsum in detsum_range:
    pedtest = eP.PedPerkeo(
        dataclass=data,
        bplot_res=False,
        bplot_fit=True,
        bplot_log=True,
        bfilt_detsum=False,
        range_detsum=[0., detsum],
    )
    print(pedtest.ret_pedestals())
    results.append(pedtest.ret_pedestals().T[0])

results = np.asarray(results).T

fig, axs = plt.subplots(4, 4, sharex=True, figsize=(8, 8))
fig.suptitle("Pedestal position for different DetSum cuts")
fig.subplots_adjust(hspace=0.6)
fig.subplots_adjust(wspace=0.6)

best_filter = []
for pmd_ind, pmt in enumerate(results):
    i_min = np.argmin(pmt)
    best_filter.append(detsum_range[i_min])
    axs.flat[pmd_ind].plot(detsum_range, pmt)
    axs.flat[pmd_ind].set_title(f"PMT{pmd_ind}")
    axs.flat[pmd_ind].set(xlabel="DetSum lowlim [ch]", ylabel="Ped [ch]")
plt.show()

print(best_filter)

"""
Do this

Find pedestal postions p

For Pedestals on det 0
Create two spectras:
DetSum
    a) Detector == 1 and PMT[i] >= (p_i + psig * 1.177) for i range(0, 8)
    b) Detector == 1 and PMT[i] < (p_i + psig * 1.177) for i range(0, 8)
    
h_b / (h_a + h_b)

Should yield ideal range for DetSum cut
"""