""""""

import numpy as np
import matplotlib.pyplot as plt
import configparser
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.config import conf_path
from panter.core.dataloaderPerkeo import DLPerkeo

cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")
ped_hist_par = {
    "bin_count": int(cnf["dataPerkeo"]["PED_hist_counts"]),
    "low_lim": int(cnf["dataPerkeo"]["PED_hist_min"]),
    "up_lim": int(cnf["dataPerkeo"]["PED_hist_max"]),
}


def pedestal_analysis(dirname: str, params: list):
    """"""

    dataloader = DLPerkeo(dirname)
    dataloader.auto()
    batch = dataloader.ret_filt_meas(["src"], [5])[0]

    data = dP.RootPerkeo(batch.file_list[0])
    data.info()

    results = []
    for detsum in range(1000, 20000, 50):
        data.set_filtdef()
        data.datafilter["Detector"].active = True
        data.datafilter["Detector"].rightval = 1
        data.datafilter["DetSum"].active = True
        data.datafilter["DetSum"].upperlimit = detsum
        data.datafilter["DetSum"].lowerlimit = 0.0
        data.auto(1)

        pedhist = dP.HistPerkeo(data.pmt_data[0], **ped_hist_par)

        pedhist.hist["y"] = np.log(pedhist.hist["y"])
        pedhist.hist["err"] = 1.0 / pedhist.hist["err"]

        fit_range = [params[0], params[1]]
        fitclass = eP.DoFit(pedhist.hist)
        fitclass.setup(eFS.pol2)
        fitclass.limitrange([fit_range[0], fit_range[1]])
        # fitclass.set_bool("boutput", True)
        fitclass.plotrange["x"] = [-150, 350]
        fitclass.plotrange["y"] = [-0.5, 10.5]
        fitres = fitclass.fit()

        results.append(
            np.asarray(
                [
                    detsum,
                    fitres.params["c0"].value,
                    fitres.params["c0"].stderr,
                    fitclass.ret_gof()["rChi2"],
                ]
            )
        )
    return np.asarray(results)


dirname = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
results = pedestal_analysis(dirname=dirname, params=[-70, 40])

print(results)

plt.plot(results.T[0], results.T[1])
plt.show()
plt.plot(results.T[0], results.T[2])
plt.show()
plt.plot(results.T[0], results.T[3])
plt.show()