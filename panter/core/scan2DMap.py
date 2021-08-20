""""""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from panter.config.filesScanMaps import scan_200117
# from panter.core.pedPerkeo import PedPerkeo
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo
import panter.config.evalFitSettings as eFS
from panter.core.evalPerkeo import DoFit
from panter import output_path


class ScanMapClass:
    """"""

    def __init__(
        self,
        scan_pos_arr: np.array,
        event_arr: np.array,
        detector: int = 0,
        mid_pos: np.array = np.array([170, 5770]),
    ):
        self._scan_pos_arr = scan_pos_arr
        self._event_arr = event_arr
        self._detector = detector
        self._mid_pos = mid_pos

        self._dataloader = DLPerkeo("")
        self._dataloader.fill(self._event_arr)
        self._meas = self._dataloader.ret_meas()

        self._pedestals = None
        self._num_pmt = 16
        self._peak_pos_map = None

    def _calc_pedestals(self):
        """Calculate pedestals for all files to be reused"""
        # FIXME: CorrPerkeo currently does not support tuples as ped_arr input
        # i.e. it does not work for type=1 events with sep. sig and bg files
        pass

    def calc_peak_positions(self, weights: np.array = None):
        """Calculate Sn peaks for all positions"""

        # if self._pedestals is None:
        # self._calc_pedestals()

        self._weights = weights
        if self._weights is None:
            self._weights = np.ones(self._num_pmt)

        self._peak_pos_map = []

        for ind, meas in enumerate(self._meas):
            # ped = self._pedestals[ind]
            ped = None
            corr_class = CorrPerkeo(meas, mode=1, ped_arr=ped, weight_arr=self._weights)
            corr_class.set_all_corr(bactive=True)
            corr_class.corrections["Drift"] = False
            corr_class.corr(bstore=True, bwrite=False)

            histp = corr_class.histograms[0]
            test = histp[1][self._detector]

            fitclass = DoFit(test.hist)
            fitclass.setup(eFS.gaus_gen)
            fitclass.set_bool("boutput", False)
            # fitclass.limit_range([8000,12000])
            fitclass.set_fitparam("mu", 10000.0)
            fitclass.fit()

            peak = fitclass.ret_results().params["mu"].value
            self._peak_pos_map.append(peak)

        self._peak_pos_map = np.asarray(self._peak_pos_map)

    def calc_loss(self, bsymm_loss: bool = True, beta_symm_loss: float = 0.5):
        """Calculate average deviation and loss over map"""

        assert self._peak_pos_map is not None, "ERROR: Map is empty."
        avg_dev = 0.0
        loss = 0.0

        mid_ind = self._find_closest_ind(self._scan_pos_arr, self._mid_pos)
        mid_peak_pos = self._peak_pos_map[mid_ind]

        for peak in self._peak_pos_map:
            avg_dev += (peak - mid_peak_pos) ** 2
        avg_dev = avg_dev / self._peak_pos_map.shape[0]

        if bsymm_loss:
            loss_lr = 0.0
            loss_ud = 0.0
            all_l, all_r = self._get_lr_pairs(self._scan_pos_arr)
            peaks_l = self._peak_pos_map[all_l]
            peaks_r = self._peak_pos_map[all_r]

            for ind in range(peaks_l.shape[0]):
                loss_lr += (peaks_l[ind] - peaks_r[ind]) ** 2
            loss_lr = loss_lr / all_l.shape[0]

            for ind in range(int(peaks_l.shape[0] * 0.5)):
                loss_ud += (peaks_l[ind] - peaks_l[peaks_l.shape[0] - 1 - ind]) ** 2
                loss_ud += (peaks_r[ind] - peaks_r[peaks_r.shape[0] - 1 - ind]) ** 2
            loss_ud = loss_ud / (int(peaks_l.shape[0] * 0.5) * 2.)

            loss += beta_symm_loss * (loss_ud + loss_lr)
        loss += loss + avg_dev

        self.avg_dev = avg_dev
        self.loss = loss

        return avg_dev, loss

    def plot_scanmap(self, bsavefig: bool = False, filename: str = ""):
        """Make a 2D plot of the scan map results."""

        det_label = f"Detector: {self._detector}/1\n"

        x = np.unique(self._scan_pos_arr.T[0])
        y = np.unique(self._scan_pos_arr.T[1])
        listx = np.arange(x.shape[0])
        listy = np.arange(y.shape[0])

        y = y[::-1]
        mappingx = dict(zip(x, listx))
        mappingy = dict(zip(y, listy))
        data = np.zeros(shape=(y.shape[0], x.shape[0]))

        for i in range(self._scan_pos_arr.shape[0]):
            indx = self._scan_pos_arr[i][0]
            indx = mappingx[indx]
            indy = self._scan_pos_arr[i][1]
            indy = mappingy[indy]
            data[indy][indx] = f"{self._peak_pos_map[i]:.0f}"

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, cmap="Wistia")

        ax.set_xticks(np.arange(x.shape[0]))
        ax.set_yticks(np.arange(y.shape[0]))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)

        string = "Weights:\n"
        string = det_label + "\n" + string
        w_rng = [self._detector * 8, len(self._weights) - 8 * (1 - self._detector)]
        for i in range(w_rng[0], w_rng[1]):
            string += f"pmt{i}/15: {self._weights[i]:.4f}\n"
        plt.text(0.02, 0.3, string, fontsize=10, transform=plt.gcf().transFigure)
        plt.setp(ax.get_xticklabels(), ha="left")

        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                ax.text(j, i, data[i, j], ha="center", va="center", color="black")

        ax.set_title(f"Scan_map with avg_dev: {self.avg_dev:.0f}")
        fig.tight_layout()

        if bsavefig:
            if filename == "":
                filename = "scan2Dmap"
            plt.savefig(f"{output_path}/{filename}.png", dpi=300)

        plt.show()

    @staticmethod
    def _find_closest_ind(all_pos_arr, target_pos):
        """"""

        assert all_pos_arr.shape[1] == target_pos.shape[0]

        dev = 0.0
        for dim in range(all_pos_arr.shape[1]):
            dev += (all_pos_arr.T[dim] - target_pos[dim]) ** 2
        dev = np.sqrt(dev)

        assert dev.min() < 50.0, "ERROR: Could not find (vague) target position."
        mid_ind = np.argmin(dev)

        return mid_ind

    @staticmethod
    def _get_lr_pairs(pos_arr: np.array):
        """"""

        x_points = np.unique(pos_arr.T[0])
        x_l = x_points[0]
        x_r = x_points[-1]
        all_left = (pos_arr[:, 0] == x_l)
        all_right = (pos_arr[:, 0] == x_r)

        return all_left, all_right


def main():
    pos, evs = scan_200117()

    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        detector=1,
    )
    smc.calc_peak_positions()
    print(smc.calc_loss())
    smc.plot_scanmap()


if __name__ == "__main__":
    main()
