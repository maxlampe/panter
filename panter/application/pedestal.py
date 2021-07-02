"""Calculate pedestals for given files"""

import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
from panter.core.pedPerkeo import PedPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
filename2 = "data119886-67506_bg.root"
filename3 = "data119874-67506_2.root"
filename4 = "data120576-67534_bg.root"
filename5 = "data120564-67534_3.root"
filename6 = "data119754-67502_beam.root"

dir2 = "/mnt/sda/PerkeoDaten1920/ElecTest_20200309/Test5_Using_double_sweep_mode_diff_Ampl/"
filename7 = "data249733.root"
filename8 = "data244813.root"

dir3 = "/mnt/sda/PerkeoDaten1920/cycle201/Det200123/"
filenames_both = ["data201932.root", "data201936.root"]
filenames_delta = ["data201948.root", "data201952.root"]
filenames_all = ["data201964-70606_bg.root", "data201980-70606_3.root"]


file = dir + filename6
data = dP.RootPerkeo(file)
pedtest = PedPerkeo(
    dataclass=data,
    bplot_res=False,
    bplot_fit=False,
    bplot_log=True,
)
print(pedtest.ret_pedestals().T[0].sum())
print(np.sqrt((pedtest.ret_pedestals().T[1] ** 2).sum()))

if True:
    step_size = 1000
    results = []
    detsum_range = range(1000, 45000, step_size)
    for detsum in detsum_range:
        pedtest = PedPerkeo(
            dataclass=data,
            bplot_res=False,
            bplot_fit=False,
            bplot_log=True,
            range_detsum=[detsum, detsum + step_size],
        )
        results.append(pedtest.ret_pedestals().T)

    results = np.asarray(results).T

    fig, axs = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("Pedestal position for different DetSum interval cuts")
    fig.subplots_adjust(hspace=0.6)
    fig.subplots_adjust(wspace=0.6)

    for pmd_ind, pmt_val in enumerate(results):
        axs.flat[pmd_ind].errorbar(detsum_range, pmt_val[0], yerr=pmt_val[1], fmt=".")
        axs.flat[pmd_ind].set_title(f"PMT{pmd_ind}")
        axs.flat[pmd_ind].set(xlabel="DetSum interval center [ch]", ylabel="Ped [ch]")
        axs.flat[pmd_ind].set_ylim([pmt_val[0][0] - 15, pmt_val[0][0] + 15])

    plt.tight_layout()
    plt.savefig("../output/pedestal_detsum_intcuts.png")
    plt.show()
