"""Studying effect of pedestal fit range and individual PMT cuts"""

import panter.core.dataPerkeo as dP
from panter.core.pedPerkeo import PedPerkeo

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

PED_CUT = 0.4

pedtest = PedPerkeo(
    dataclass=data,
    bplot_res=False,
    bplot_fit=True,
    bplot_log=False,
)
ped = pedtest.ret_pedestals().T[0]
sig = pedtest.ret_pedestals().T[2]

#
# Get all events
#

data.set_filtdef()
data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
data.auto(1)
data.gen_hist(
    lpmt=range(0, data.no_pmts),
    cust_histsum_par={"bin_count": 500, "low_lim": -500, "up_lim": 5000},
)

hist_all = data.hist_sums[0]

#
# Get only back scattered events + half of pedestal
#

data.set_filtdef()
data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
coll_cut = 0
for no_pmt, pmt_ped in enumerate(ped):
    if no_pmt > 7:
        continue
    data.set_filt(
        "data",
        fkey="PMT",
        active=True,
        ftype="num",
        low_lim=pmt_ped,
        up_lim=80e3,
        index=no_pmt,
    )
print(f"Collected cut: {coll_cut}")
data.auto(1)
data.gen_hist(
    lpmt=range(0, data.no_pmts),
    cust_histsum_par={"bin_count": 500, "low_lim": -500, "up_lim": 5000},
)
hist_pedback = data.hist_sums[0]

#
# Get only back scattered events
#

data.set_filtdef()
data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
coll_cut = 0
for no_pmt, pmt_ped in enumerate(ped):
    if no_pmt > 7:
        continue
    coll_cut += pmt_ped + PED_CUT * sig[no_pmt]
    data.set_filt(
        "data",
        fkey="PMT",
        active=True,
        ftype="num",
        low_lim=pmt_ped + PED_CUT * sig[no_pmt],
        up_lim=80e3,
        index=no_pmt,
    )
print(f"Collected cut: {coll_cut}")
data.auto(1)
data.gen_hist(
    lpmt=range(0, data.no_pmts),
    cust_histsum_par={"bin_count": 500, "low_lim": -500, "up_lim": 5000},
)
hist_onlyback = data.hist_sums[0]

hist_onlyback.divbyhist(hist_all)
hist_pedback.divbyhist(hist_all)
hist_onlyback.plot_hist(
    title="Backscat / All (PMT0-7)",
    xlabel="ADC [ch]",
    ylabel="Ratio [ ]",
    bsavefig=True,
    filename=f"Pedestal_BackscatOverAll_PED_CUT{PED_CUT}",
)
hist_pedback.plot_hist(
    title="(Backscat + Ped) / All (PMT0-7)",
    xlabel="ADC [ch]",
    ylabel="Ratio [ ]",
    bsavefig=True,
    filename=f"Pedestal_PedAndBackscatOverAll_PED_CUT{PED_CUT}",
)
