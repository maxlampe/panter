""""""

from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import corrPerkeo

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

meas = dataloader.ret_filt_meas(["cyc_no"], [194070])[0]
print(meas())

file = meas()[2]
print(file)

corr_class = corrPerkeo(meas)
corr_class.corrections["Pedestal"] = False
corr_class.corrections["RateDepElec"] = False
# corr_class.corr()

import panter.core.dataPerkeo as dP

data = dP.RootPerkeo(file[0])
data.auto()


exit()
