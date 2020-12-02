"""Pedestal analysis"""

import dataPerkeo as dP
import evalPerkeo as eP

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
filename2 = "data119886-67506_bg.root"
filename3 = "data119874-67506_2.root"
filename4 = "data120576-67534_bg.root"
filename5 = "data120564-67534_3.root"
filename6 = "data119754-67502_beam.root"

dir2 = (
    "/mnt/sda/PerkeoDaten1920/ElecTest_20200309/Test5_Using_double_sweep_mode_diff_Ampl"
)
filename5 = "data249733.root"
filename6 = "data244813.root"

file = dir + filename4
data = dP.RootPerkeo(file)

pedtest = eP.PedPerkeo(data)
print(pedtest.ret_pedestals())
pedtest.plot_pedestals()
