"""Dump - For testing and staging"""

import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import corrPerkeo

# General import parameters
file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
filename = "data119754-67502_beam.root"

# Direct access to raw root file
data = dP.RootPerkeo(file_dir + filename)
data.info()
data.auto()
data.gen_hist(data.ret_actpmt())
data.hists[2].plt()

# Using the data loader and the corrPerkeo class to generate a fully corrected spectra
dataloader = DLPerkeo(file_dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 67502])

corr_class = corrPerkeo(dataloader=filt_meas, mode=0)
corr_class.set_all_corr(bactive=True)
corr_class.corr(bstore=True, bwrite=False)

corr_class.histograms[0][1][0].plt()
