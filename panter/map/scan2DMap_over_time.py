"""Create 2D scan correction map from precalculated results."""

import datetime
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panter.base.mapPerkeo import MapPerkeo
from panter.config import conf_path
from panter.data.dataMisc import FilePerkeo
import panter.config.filesScanMaps as filesScanMaps

output_path = os.getcwd()

MAPS = [
    filesScanMaps.scan_200112,
    filesScanMaps.scan_200113,
    filesScanMaps.scan_200114,
    # filesScanMaps.scan_200115,
    filesScanMaps.scan_200116_3,
    filesScanMaps.scan_200117,
    filesScanMaps.scan_200118,
    filesScanMaps.scan_200119,
    filesScanMaps.scan_200120,
    filesScanMaps.scan_200121,
    filesScanMaps.scan_200122,
]

DET0 = [
    None,
    None,
    [1.020390, 1.001549, 0.951786, 0.980865, 0.988492, 1.000348, 0.996493, 1.003737],
    # None,
    [0.986570, 1.003502, 0.946171, 0.984044, 0.999282, 1.002169, 0.998284, 0.987721],
    [1.004935, 0.996130, 0.953226, 0.974597, 1.001476, 0.991785, 1.000493, 0.992223],
    [1.013583, 1.004575, 0.958183, 0.972812, 0.986177, 0.989460, 0.997237, 0.986930],
    [1.016379, 0.992181, 0.952013, 0.973492, 0.991813, 0.993097, 0.997753, 0.994426],
    [0.981491, 1.000165, 0.940416, 0.972391, 1.009283, 0.998123, 1.007426, 1.001705],
    [0.984937, 0.994228, 0.950598, 0.972297, 1.009479, 0.996995, 1.000693, 0.996841],
    [0.977252, 1.000225, 0.951456, 0.966188, 1.002773, 0.998498, 1.008772, 0.986460],
]

DET1 = [
    None,
    None,
    None,
    # None,
    [0.959239, 0.943674, 0.968899, 0.953212, 1.018763, 0.987511, 1.006586, 1.005496],
    [0.933810, 0.911781, 0.948395, 0.943526, 1.039428, 1.002044, 1.015895, 1.021173],
    None,
    [0.925773, 0.920650, 0.950969, 0.945447, 1.042692, 0.998446, 1.017786, 1.014351],
    [0.929096, 0.915618, 0.948542, 0.943670, 1.033680, 0.997584, 1.016141, 1.015991],
    [0.922084, 0.931532, 0.951370, 0.942273, 1.036510, 0.996205, 1.014771, 1.017764],
    [0.929315, 0.927203, 0.948378, 0.956256, 1.031227, 0.996117, 1.011635, 1.015677],
]

assert len(MAPS) == len(DET0) and len(MAPS) == len(DET1), "ERROR: Inputs not right."


class ScanFac2DMap(MapPerkeo):
    """"""

    def __init__(
        self,
        bimp_2dfac: bool = True,
    ):
        super().__init__(fmeas=np.asarray([]), level=1, bimport=[bimp_2dfac])
        self._outfile = ["det_fac_2D_map.p"]

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""

        if level == 0 and bimp:
            # try to import pmt factor map
            impfile = FilePerkeo(f"{conf_path}/{self._outfile[0]}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[level].shape[0] > 0, "ERROR: PMT 2D factor map empty."

            return True
        else:
            self.maps[0] = pd.DataFrame(columns=["time", "pmt_fac"])
            self._set_pmt2d_fac()
            return False

    def _set_pmt2d_fac(self, bfill_to_nearest: bool = True):
        """Set 2D correction factors for each PMT from scan optimization."""

        for index, map_files in enumerate(MAPS):

            pos, evs = map_files()
            time = os.path.getmtime(evs[0][2][0])

            fac0 = DET0[index]
            fac1 = DET1[index]
            if fac0 is None:
                fac0 = [None] * 8
            if fac1 is None:
                fac1 = [None] * 8
            factors = np.array(fac0 + fac1)

            pmt_dict = {
                "time": time,
                "pmt_fac": factors,
            }

            self.maps[0] = self.maps[0].append(pmt_dict, ignore_index=True)

        if bfill_to_nearest:
            print(self.maps[0])
            self.maps[0] = self._fill_with_closest(self.maps[0])
            print(self.maps[0])

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile[0]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def plot_pmt2d_map(self, bsave: bool = False):
        """Plot pmt 2D correction factor map for both detectors"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("PMT 2D scan correction factors")

        axs[0].set_title("2D Factor over time Det 0")
        axs[1].set_title("2D Factor over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="2D Correction factor [ ]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="2D Correction factor [ ]")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        axs[0].xaxis.set_major_formatter(xfmt)
        axs[1].xaxis.set_major_formatter(xfmt)

        pmt_fac = self.maps[0]["pmt_fac"].apply(pd.Series)
        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

        for PMT in range(8):
            axs[0].plot(
                dates_plot,
                pmt_fac[PMT],
                "-x",
                label=f"PMT{PMT}",
            )
            axs[1].plot(
                dates_plot,
                pmt_fac[PMT + 8],
                "-x",
                label=f"PMT{PMT + 8}",
            )
        axs[0].legend()
        axs[1].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[0][:-1] + "png", dpi=300)
        plt.show()

        return 0

    @staticmethod
    def _fill_with_closest(fac_map: pd.DataFrame):
        """"""

        times = fac_map["time"].apply(pd.Series)
        facs = fac_map["pmt_fac"].apply(pd.Series)

        t_dist_mat = []
        for i in range(times.shape[0]):
            t_curr = times[0][i]
            t_curr_diff = np.abs(times[0] - t_curr)
            t_sorted = t_curr_diff.sort_values()
            t_dist_mat.append(list(t_sorted.index.values[1:]))

        for pmt in range(16):
            inter_fac = list(facs[pmt])
            for j, fac in enumerate(facs[pmt]):
                if pd.isnull(fac):
                    while True:
                        for next_t in t_dist_mat[j]:
                            if not pd.isnull(facs[pmt][next_t]):
                                inter_fac[j] = facs[pmt][next_t]
                                break
                        break
            facs[pmt] = inter_fac

        new_map = pd.DataFrame(columns=["time", "pmt_fac"])
        for i in range(times.shape[0]):
            pmt_dict = {
                "time": times[0][i],
                "pmt_fac": list(facs.T[i]),
            }
            new_map = new_map.append(pmt_dict, ignore_index=True)

        return new_map


def main():
    sf2Dm = ScanFac2DMap(bimp_2dfac=True)
    sf2Dm()
    sf2Dm.plot_pmt2d_map()


if __name__ == "__main__":
    main()
