"""Comparing pedestal results for different sources"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
from ped_map import PedMapPerkeo

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

PMT = 1
dates = np.asarray(
    [
        # Allmode
        datetime.datetime(2020, 1, 19, 18, 10).timestamp(),
        # Bothmode
        datetime.datetime(2020, 1, 20, 7, 10).timestamp(),
        # Allmode
        datetime.datetime(2020, 1, 23, 12, 30).timestamp(),
        # Bothmode
        datetime.datetime(2020, 1, 23, 20, 45).timestamp(),
    ]
)


sources = np.asarray(["beam", "ce", "cd", "cs", "bi", "sn"])

fig, axs = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("Pedestal Map")

axs[0].set_title("Pedestals over time Det 0")
axs[1].set_title("Peak pos over time Det 1")
axs.flat[0].set(xlabel="Time [s]", ylabel="Ped pos [ch]")
axs.flat[1].set(xlabel="Time [s]", ylabel="Ped pos [ch]")
xfmt = md.DateFormatter("%m-%d\n%H:%M")
axs[0].xaxis.set_major_formatter(xfmt)
axs[1].xaxis.set_major_formatter(xfmt)
plt.tight_layout()

dataf = pd.DataFrame()
for src in sources:
    ppm = PedMapPerkeo([], bimp_ped=True, outfile_flag=src)
    ppm()

    # df_filt = ppm.maps[0].query("1579000000 < time < 1579100000")
    # df_filt = ppm.maps[0][ppm.maps[0]["time"] > dates[1]]
    # df_filt = df_filt[df_filt["time"] < dates[2]]

    df_filt = ppm.maps[0][ppm.maps[0]["time"] < dates[0]]

    ped_df = df_filt["ped_list"].apply(pd.Series)
    ped_err_df = df_filt["ped_err_list"].apply(pd.Series)

    # print((ped_err_df == 0).sum())
    dataf[src] = (ped_df / ped_err_df ** 2).sum() / (1.0 / ped_err_df ** 2).sum()
    dataf[src + "_err"] = np.sqrt(1.0 / (1.0 / ped_err_df ** 2).sum())
    dates_plot = [datetime.datetime.fromtimestamp(t) for t in df_filt["time"]]

    axs[0].errorbar(
        dates_plot,
        ped_df[PMT],
        yerr=ped_err_df[PMT],
        fmt=".",
        label=f"PMT{PMT}_{src}",
    )
    axs[1].errorbar(
        dates_plot,
        ped_df[PMT + 8],
        yerr=ped_err_df[PMT + 8],
        fmt=".",
        label=f"PMT{PMT + 8}_{src}",
    )

axs[0].legend()
axs[1].legend()
plt.show()


dataf_diff = pd.DataFrame()
for src in sources[1:]:
    dataf_diff[f"bm_{src}"] = dataf["beam"] - dataf[src]
    dataf_diff[f"bm_{src}_err"] = np.sqrt(
        (dataf["beam_err"] ** 2 + dataf[f"{src}_err"] ** 2)
    )

print("\nWAM Pedestals for different sources for each PMT:\n")
print(dataf)
print("\nShift of pedestal compared to beam for each PMT:\n")
print(dataf_diff)
print("\nShift on DetSum compared to beam pedestal:\n")
print(dataf_diff.sum())
