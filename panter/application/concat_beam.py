"""Concatenate all or several measurements with corrections into a histogram."""

from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])

corr_class = CorrPerkeo(dataloader=filt_meas, mode=0)
corr_class.set_all_corr(bactive=True)

corr_class.addition_filters.append(
    {
        "tree": "data",
        "fkey": "Detector",
        "active": True,
        "ftype": "bool",
        "rightval": 0,
    }
)

corr_class.corr(bstore=True, bwrite=False, bconcat=True)

corr_class.hist_concat.plt()
corr_class.hist_concat.write2root(histname=f"DetSumTot", filename="concat_test_raw")
