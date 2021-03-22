"""Comparing pedestal results for different sources"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ped_map import PedMapPerkeo

PMT = 1

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Pedestal Map")

axs[0].set_title("Pedestals over time Det 0")
axs[1].set_title("Peak pos over time Det 1")
axs.flat[0].set(xlabel="Time [s]", ylabel="Ped pos [ch]")
axs.flat[1].set(xlabel="Time [s]", ylabel="Ped pos [ch]")

dataf = pd.DataFrame()
for src in ["beam", "ce", "cd", "cs", "bi", "sn"]:
    ppm = PedMapPerkeo([], bimp_ped=True, outfile_flag=src)
    ppm()

    df_filt = ppm.maps[0].query("1579000000 < time < 1579100000")

    ped_df = df_filt["ped_list"].apply(pd.Series)
    ped_err_df = df_filt["ped_err_list"].apply(pd.Series)

    print((ped_err_df == 0).sum())

    dataf[src] = (ped_df / ped_err_df ** 2).sum() / (1.0 / ped_err_df ** 2).sum()
    dataf[src + "_err"] = np.sqrt(1. / (1.0 / ped_err_df ** 2).sum())
    # print(ped_df)

    axs[0].errorbar(
        df_filt["time"],
        ped_df[PMT],
        yerr=ped_err_df[PMT],
        fmt=".",
        label=f"PMT{PMT}_{src}",
    )
    axs[1].errorbar(
        df_filt["time"],
        ped_df[PMT + 8],
        yerr=ped_err_df[PMT + 8],
        fmt=".",
        label=f"PMT{PMT + 8}_{src}",
    )
axs[0].legend()
axs[1].legend()

print(dataf)

plt.show()


