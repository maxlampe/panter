"""Dump - For testing and staging"""

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

pedtest = eP.PedPerkeo(
    dataclass=data,
    bplot_res=False,
    bplot_fit=False,
    bplot_log=True,
)
ped = pedtest.ret_pedestals().T[0]
sig = pedtest.ret_pedestals().T[2]
print(ped)

#
# Get all events
#

data.set_filtdef()
data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
data.auto(1)
data.gen_hist(
    lpmt=range(0, data.no_pmts),
    cust_histsum_par={"bin_count": 500, "low_lim": -500, "up_lim": 10000},
)

hist_all = data.hist_sums[2]
hist_all.plt()

#
# Get only back scattered events
#

data.set_filtdef()
data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
coll_cut = 0
for no_pmt, pmt_ped in enumerate(ped):
    if no_pmt > 7:
        continue
    coll_cut += pmt_ped + 1.177 * sig[no_pmt]
    data.set_filt(
        "data",
        fkey="PMT",
        active=True,
        ftype="num",
        low_lim=pmt_ped + 1.177 * sig[no_pmt],
        up_lim=80e3,
        index=no_pmt,
    )
print(f"Collected cut: {coll_cut}")
data.auto(1)
data.gen_hist(
    lpmt=range(0, data.no_pmts),
    cust_histsum_par={"bin_count": 500, "low_lim": -500, "up_lim": 10000},
)
# data.hist_sums[0].plt()
# data.hist_sums[1].plt()
# data.hist_sums[2].plt()
hist_onlyback = data.hist_sums[2]
hist_onlyback.plt()

# hist_all.addhist(hist_onlyback)

hist_onlyback.divbyhist(hist_all)
hist_onlyback.plt()

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
