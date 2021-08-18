"""Studying effect of pedestal position to DTT cuts"""

import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
from panter.core.pedPerkeo import PedPerkeo
from panter import output_path

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

DTT_STEP = 50000

ped_values = [[] for i in range(16)]
ped_values_err = [[] for i in range(16)]
dtt_vals = []

pedtest = PedPerkeo(
    dataclass=data,
    bplot_res=False,
    bplot_fit=False,
    bplot_log=False,
)
def_ped = pedtest.ret_pedestals().T[0]
def_ped_err = pedtest.ret_pedestals().T[1]

for delt_dtt in range(0, 1200000, DTT_STEP):
    pedtest = PedPerkeo(
        dataclass=data,
        bplot_res=False,
        bplot_fit=False,
        bplot_log=False,
        range_dtt=[delt_dtt, delt_dtt + DTT_STEP],
    )
    ped = pedtest.ret_pedestals().T[0]
    ped_err = pedtest.ret_pedestals().T[1]

    dtt_vals.append((delt_dtt + DTT_STEP * 0.5) * 0.01)
    for i in range(16):
        ped_values[i].append(ped[i])
        ped_values_err[i].append(ped_err[i])

print(ped_values)

fig, axs = plt.subplots(4, 2, figsize=(15, 15))
fig.suptitle("DTT Pedestal cuts")
fig.subplots_adjust(hspace=0.5)

for i in range(8):
    c0 = def_ped[i]
    c0_e = def_ped_err[i]
    axs.flat[i].errorbar(dtt_vals, ped_values[i], yerr=ped_values_err[i], fmt=".")
    axs.flat[i].set_xlabel("DeltaTriggerTime [mus]")
    axs.flat[i].set_ylabel(f"Ped_PMT{i} [ch]")
    axs.flat[i].plot(dtt_vals, [c0] * len(dtt_vals))
    axs.flat[i].fill_between(
        dtt_vals, [c0 - c0_e] * len(dtt_vals), [c0 + c0_e] * len(dtt_vals), color="r"
    )

    axs.flat[i].annotate(
        f"Red/Or: Ped for all DTT with fit error (default)",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="1"),
    )
plt.savefig(output_path + "/" + "ped_dtt_cuts.png", dpi=300)
plt.show()
