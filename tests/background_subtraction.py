""""""

import os
import subprocess
import numpy as np
import pandas as pd
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import corrPerkeo
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


FIT_RANGE = [40000, 51500]
MIN_ACC = 0.01

dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(dir)
dataloader.auto()

batches = dataloader.ret_filt_meas(["src"], [5])[900:940]
res_df = pd.DataFrame(
    columns=["r_c0", "r_c0_err", "r_rCh2", "p_c0", "p_c0_err", "p_rCh2", "rel_dev"],
    index=range(len(batches)),
)

print(batches)

for ind, meas in enumerate(batches):
    # do root analysis
    file = meas()[2][0]
    print(file)

    root_cmd = "/home/max/Software/root_install/bin/root"
    this_path = os.path.dirname(os.path.realpath(__file__))
    arg_old = (
        f"{this_path}/background_subtraction.cpp"
        + f'("{file}", {FIT_RANGE[0]}, {FIT_RANGE[1]})'
    )
    subprocess.run([root_cmd, arg_old])

    root_fitres = open("root_fitres.txt").read().split()
    root_fitres = list(map(float, root_fitres))
    subprocess.run(["rm", "root_fitres.txt"])

    # do panter analysis
    corr_class = corrPerkeo(meas)
    corr_class.corrections["Pedestal"] = False
    corr_class.corrections["RateDepElec"] = False
    corr_class.corr(bstore=True)

    panter_fitres = []
    for hist in corr_class.histograms[0, 0]:
        fitclass = eP.DoFit(hist.hist)
        fitclass.setup(eFS.pol0)
        fitclass.limitrange([FIT_RANGE[0], FIT_RANGE[1]])

        fitres = fitclass.fit()

        panter_fitres.append(fitres.params["c0"].value)
        panter_fitres.append(fitres.params["c0"].stderr)
        panter_fitres.append(fitclass.ret_gof()["rChi2"])

    root_fitres = np.asarray(root_fitres)
    panter_fitres = np.asarray(panter_fitres)
    rel_dev = abs((root_fitres - panter_fitres)) / root_fitres
    av_dev = rel_dev.mean()

    if av_dev <= MIN_ACC:
        print(
            f"SUCCESS: Test passed. Average deviation {av_dev} <= minimum accuracy {MIN_ACC}"
        )
    else:
        print(
            f"FAILED: Test not passed. Average deviation {av_dev} > minimum accuracy {MIN_ACC}"
        )
    print(f"Relative deviations:\n{rel_dev}")

    r_c0 = [root_fitres[0], root_fitres[3]]
    r_c0_err = [root_fitres[1], root_fitres[4]]
    r_rCh2 = [root_fitres[2], root_fitres[5]]
    p_c0 = [panter_fitres[0], panter_fitres[3]]
    p_c0_err = [panter_fitres[1], panter_fitres[4]]
    p_rCh2 = [panter_fitres[2], panter_fitres[5]]
    rel_dev = [rel_dev[:3].mean(), rel_dev[3:].mean()]
    res_dict = {
        "r_c0": r_c0,
        "r_c0_err": r_c0_err,
        "r_rCh2": r_rCh2,
        "p_c0": p_c0,
        "p_c0_err": p_c0_err,
        "p_rCh2": p_rCh2,
        "rel_dev": rel_dev,
    }
    res_df.loc[ind] = pd.Series(res_dict)

print(res_df)
