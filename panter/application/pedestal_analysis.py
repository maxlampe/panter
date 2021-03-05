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
    """Conduct pedestal calculations with different cuts and parameter changes."""

    dataloader = DLPerkeo(dirname)
    dataloader.auto()
    batch = dataloader.ret_filt_meas(["src"], [5])[0]

    data = dP.RootPerkeo(batch.file_list[0])
    data.info()

    results = []
    for detsum in range(1000, 40000, 20000):
        data.set_filtdef()
        data.set_filt("data", fkey="Detector", active=True, ftype="bool", rightval=1)
        data.set_filt(
            "data",
            fkey="DetSum",
            active=True,
            ftype="num",
            low_lim=detsum,
            up_lim=100e3,
        )
        data.auto(1)

        pedhist = dP.HistPerkeo(data.pmt_data[params[2]], **ped_hist_par)

        pedloghist = pedhist

        pedloghist.hist["y"] = np.log(pedloghist.hist["y"])
        pedloghist.hist["err"] = 1.0 / pedloghist.hist["err"]

        fit_range = [params[0], params[1]]
        fitclass = eP.DoFit(pedloghist.hist)
        fitclass.setup(eFS.pol2)
        fitclass.limit_range([fit_range[0], fit_range[1]])
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
                    fitclass.ret_gof()[0],
                    fitclass.ret_gof()[1],
                ]
            )
        )
    return np.asarray(results)


dirname = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"

res = []
for pmt in [8]:
    results = pedestal_analysis(dirname=dirname, params=[-70, 40, pmt])

    i_max = np.argmin(results.T[3])
    res.append([pmt, results.T[0, i_max], results.T[3, i_max], results.T[4, i_max]])

print(res)
