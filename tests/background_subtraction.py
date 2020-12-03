""""""

from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import corrPerkeo
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

meas = dataloader.ret_filt_meas(["cyc_no"], [194070])

corr_class = corrPerkeo(meas)
corr_class.corrections["Pedestal"] = False
corr_class.corrections["RateDepElec"] = False
corr_class.corr(bstore=True)

for hist in corr_class.histograms[0, 0]:
    fitclass = eP.DoFit(hist.hist)
    fitclass.setup(eFS.pol0)
    fitclass.limitrange([45000.0, 50000.0])

    fitres = fitclass.fit()
    print(fitres.fit_report())

exit()

import panter.core.dataPerkeo as dP

file = meas[0]()[2]
print(file)
data = dP.RootPerkeo(file[0])
data.auto()
