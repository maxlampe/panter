"""Example file of main panter classes.

It is recommended to check docstrings for additional information and examples."""

import numpy as np
import panter.core.dataPerkeo as dP
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import corrPerkeo

# General import parameters
file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"

#
# Base histogram class HistPerkeo
#
if True:
    hist1 = dP.HistPerkeo(
        data=np.array([0, 0, 1, 2, 3, 4, 5]), bin_count=10, low_lim=-10, up_lim=10
    )
    hist2 = dP.HistPerkeo(
        data=np.array([0, 0, -1, 0, -3, 2, -1]), bin_count=10, low_lim=-10, up_lim=10
    )
    hist1.plt()
    hist2.plt()
    hist1.addhist(hist2)
    hist1.plt()

#
# Perkero root file core management class RootPerkeo for raw access
#

if True:
    filename = "data119754-67502_beam.root"

    data = dP.RootPerkeo(file_dir + filename)
    data.info()
    data.auto()
    data.gen_hist(data.ret_actpmt())
    data.hists[2].plt()

#
# Data loader for automated file fetching and loading if needed
#

if True:
    # Using the data loader and corrPerkeo class to generate a fully corrected spectra
    dataloader = DLPerkeo(file_dir)
    dataloader.auto()
    filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 67502])

    #
    # Core data reduction and correction class corrPerkeo
    #

    corr_class = corrPerkeo(dataloader=filt_meas, mode=0)
    corr_class.set_all_corr(bactive=True)
    corr_class.corr(bstore=True, bwrite=False)

    corr_class.histograms[0][1][0].plt()
