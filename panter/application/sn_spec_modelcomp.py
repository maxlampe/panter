""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
from panter.core.mapPerkeo import MapPerkeo
from panter.config.evalFitSettings import gaus_expmod, charge_spec
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo
from panter.config import conf_path
from panter import output_path


if False:
    x = np.linspace(0.0, 5000.0, num=1000)
    """
    params = {
        "w": 0.3,
        "a": 0.03,
        "lam": 1.4,
        "q0": 2.0,
        "sig0": 0.9,
        "c0": 10.0,
        "sig": 3.0,
        "mu": 5.0,
        "k_max": 100,
        "norm": 5000000.
    }
    """
    params = {
        "w": 0.9993,
        "a": 0.03,
        "lam": 1.4,
        "q0": 2.0,
        "sig0": 0.9,
        "c0": 10.0,
        "sig": 3.0,
        "mu": 5.0,
        "k_max": 100,
        "norm": 5000000.0,
    }

    val = charge_spec(x, **params)
    plt.plot(x, val)
    plt.show()

    exit()

file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(file_dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 3])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])

res_df = pd.DataFrame(
    columns=["time", "mu_val", "lam_val", "q0_val", "sig0_val", "c0_val", "sig_val"]
)

for i, meas in enumerate(filt_meas):
    if i in [61, 109, 294, 411, 625]:
        continue
    if i > 0:
        break

    print(f"Meas No: {i}")

    time = meas.date_list[0]

    corr_class = CorrPerkeo(dataloader=meas, mode=2)
    corr_class.set_all_corr(bactive=False)
    corr_class.corrections["DeadTime"] = True
    corr_class.corrections["Pedestal"] = True
    corr_class.corrections["RateDepElec"] = True

    corr_class.corr(bstore=True, bwrite=False)

    # fitsettings = gaus_expmod
    fitsettings = charge_spec
    fitsettings.plot_labels = [
        "SnSpec fit result",
        "ADC [ch]",
        "Counts [ ]",
    ]
    fitsettings.plotrange["x"] = [30.0, 4000.0]

    hists = corr_class.histograms[0][1]
    rchi2 = np.array([])
    mu_val = np.array([])
    lam_val = np.array([])
    q0_val = np.array([])
    sig0_val = np.array([])
    c0_val = np.array([])
    sig_val = np.array([])
    mu_err = np.array([])

    for j in range(len(hists)):
        dofitclass = eP.DoFit(hists[j].hist)
        dofitclass.setup(fitsettings)
        dofitclass.set_bool("boutput", False)
        dofitclass.fit()

        fit_results = dofitclass.ret_results()
        rchi2 = np.append(rchi2, dofitclass.ret_gof()[0])
        mu_val = np.append(mu_val, fit_results.params["mu"].value)
        lam_val = np.append(lam_val, fit_results.params["lam"].value)
        q0_val = np.append(q0_val, fit_results.params["q0"].value)
        sig0_val = np.append(sig0_val, fit_results.params["sig0"].value)
        c0_val = np.append(c0_val, fit_results.params["c0"].value)
        sig_val = np.append(sig_val, fit_results.params["sig"].value)
        """
        mu_val = np.append(mu_val, fit_results.params["mu"].value)
        mu_err = np.append(mu_err, fit_results.params["mu"].stderr)
        """

    """
    meas_dict = {
        "time": time,
        "peak_list": mu_val,
        "err_list": mu_err,
        "rchi2": rchi2,
    }
    """
    meas_dict = {
        "time": time,
        "mu_val": mu_val,
        "lam_val": lam_val,
        "q0_val": q0_val,
        "sig0_val": sig0_val,
        "c0_val": c0_val,
        "sig_val": sig_val,
    }

    res_df = res_df.append(meas_dict, ignore_index=True)


print(res_df)
