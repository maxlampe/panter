"""Class to calculate and visualize 2D scan results."""

import os

import matplotlib.pyplot as plt
import numpy as np

from panter.config.evalFitSettings import gaus_gen
from panter.config.filesScanMaps import scan_200116_3, scan_200117
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.corrPerkeo import CorrPerkeo
from panter.eval.evalFit import DoFit
from panter.eval.pedPerkeo import PedPerkeo
from panter.config import conf_path

output_path = os.getcwd()


class ScanMapClass:
    """Class to calculate and visualize 2D scan results.

    Uses pre-sorted files and position values from filesScanMaps in panter.config.

    Parameters
    ----------
    scan_pos_arr, event_arr: np.array, np.array
        Position and measurement array from filesScanMaps class.
    detector: 0
        Target detector.
    mid_pos: np.array([170, 5770])
        Center position value.
    label: "unlabelled"
        Label to be used for plot.
    mu_init_val: float
        Initial value for the Gaussian peak fit to extract a peak position.
    fit_range: list
        Fit range for the Gaussian peak fit.
    buse_2Dcorr, buse_sim_loss, do_widths: bool, bool, bool
        Set of bools to determine whether to 1) apply 2D correctionf actors, 2) use
        simulation map as ideal value and calculate deviation from that as loss, or 3)
        use widths of Gaussian peaks instead of positions.

    Attributes
    ----------
    detector
    label
    avg_dev, loss: float, float
        Average deviation and loss over the 2D map.
    """

    def __init__(
        self,
        scan_pos_arr: np.array,
        event_arr: np.array,
        detector: int = 0,
        mid_pos: np.array = np.array([170, 5770]),
        label: str = "unlabelled",
        buse_2Dcorr: bool = False,
        buse_sim_loss: bool = False,
        mu_init_val: float = None,
        fit_range: list = None,
        do_widths: bool = False,
    ):
        self._scan_pos_arr = scan_pos_arr
        self._event_arr = event_arr
        self.detector = detector
        self._mid_pos = mid_pos
        self.label = label
        self._buse_2Dcorr = buse_2Dcorr
        self._buse_sim_loss = buse_sim_loss
        self._mu_init = mu_init_val
        self._fit_range = fit_range
        self._do_widths = do_widths

        self._dataloader = DLPerkeo("")
        self._dataloader.fill(self._event_arr)
        self._meas = self._dataloader.ret_meas()

        self._pedestals = None
        self._weights = None
        self._num_pmt = 16
        self._peak_pos_map = None
        self._peak_pos_err_map = None
        self.avg_dev = None
        self.loss = None

        # Calculate middle peak positions for weights=np.ones(16)
        self._mid_ind = self._find_closest_ind(self._scan_pos_arr, self._mid_pos)
        self._center_peak = self._calc_single_peak(self._meas[self._mid_ind])

        if self._buse_sim_loss:
            sim_results = np.loadtxt(conf_path + "/scan_sim_target_sn.txt")
            self._sim_pos = sim_results[:, :2]
            self._sim_targets = sim_results[:, 2]

    def _calc_pedestals(self):
        """Calculate pedestals for all files to be reused"""

        self._pedestals = []
        for ind, meas in enumerate(self._meas):
            data_sig = RootPerkeo(meas.file_list[0])
            ped_sig = PedPerkeo(
                dataclass=data_sig,
            ).ret_pedestals()
            data_bg = RootPerkeo(meas.file_list[1])
            ped_bd = PedPerkeo(
                dataclass=data_bg,
            ).ret_pedestals()
            self._pedestals.append([ped_sig, ped_bd])

    def _calc_single_peak(
        self,
        meas,
        ped=None,
        brec: bool = True,
        mu_init_val=None,
        fit_range=None,
        bplot_fits: bool = False,
    ):
        """Calculate the position and width of a measurement with a Gaussian fit."""

        if self._do_widths:
            tar_par = "sig"
        else:
            tar_par = "mu"
        if mu_init_val is None:
            if self._mu_init is not None:
                mu_init_val = self._mu_init
            else:
                mu_init_val = 10500.0
        if ped is None:
            ped = [None, None]
        corr_class = CorrPerkeo(
            meas, mode=1, ped_arr=ped[0], bgped_arr=ped[1], weight_arr=self._weights
        )
        corr_class.set_all_corr(bactive=False)
        corr_class.corrections["Drift"] = False
        corr_class.corrections["Scan2D"] = self._buse_2Dcorr
        corr_class.corrections["RateDepElec"] = True
        corr_class.corrections["Pedestal"] = True
        corr_class.corrections["DeadTime"] = True
        corr_class.corr(bstore=True, bwrite=False)

        histp = corr_class.histograms[0]
        test = histp[1][self.detector]

        fitclass = DoFit(test.hist)
        fitclass.setup(gaus_gen)
        fitclass.set_bool("boutput", bplot_fits)
        if fit_range is not None:
            fitclass.limit_range(fit_range)
        else:
            if self._fit_range is not None:
                fitclass.limit_range(self._fit_range)
        fitclass.set_fitparam("mu", mu_init_val)
        fitclass.fit()

        try:
            peak = fitclass.ret_results().params[tar_par].value
            peak_err = fitclass.ret_results().params[tar_par].stderr
        except AttributeError:
            if brec:
                print("Trying refit with higher mu val")
                fitclass.set_fitparam("mu", mu_init_val * 1.05)
                fitclass.fit()
                try:
                    peak = fitclass.ret_results().params[tar_par].value
                    peak_err = fitclass.ret_results().params[tar_par].stderr
                except AttributeError:
                    print(meas)
                    print(self._weights)
                    test.plot_hist()
                    peak = None
                    peak_err = None
            else:
                peak = None
                peak_err = None

        if self._do_widths:
            peak = np.abs(peak)
            peak_err = np.abs(peak_err)

        return peak, peak_err

    def calc_peak_positions(
        self,
        weights: np.array = None,
        mu_init_val: float = None,
        fit_range: list = None,
        bplot_fits: bool = False,
    ):
        """Calculate Sn peaks for all positions"""

        if self._pedestals is None:
            self._calc_pedestals()

        self._weights = weights
        if self._weights is None:
            self._weights = np.ones(self._num_pmt)

        self._peak_pos_map = []
        self._peak_pos_err_map = []

        for ind, meas in enumerate(self._meas):
            ped = self._pedestals[ind]
            peak, peak_err = self._calc_single_peak(
                meas,
                ped,
                mu_init_val=mu_init_val,
                fit_range=fit_range,
                bplot_fits=bplot_fits,
            )
            self._peak_pos_map.append(peak)
            self._peak_pos_err_map.append(peak_err)

        self._peak_pos_map = np.asarray(self._peak_pos_map)
        self._peak_pos_err_map = np.asarray(self._peak_pos_err_map)

    def ret_peak_map(self):
        """Returns calculated array of peak pos values and their fit errors."""
        return (
            self._peak_pos_map,
            self._peak_pos_err_map,
            self._mid_ind,
            self._center_peak,
        )

    def calc_loss(self, bsymm_loss: bool = True, beta_symm_loss: float = 0.5):
        """Calculate the loss for the current set of peak positions.

        Parameters
        ----------
        bsymm_loss: True
            Add the symmetry loss to the uniformity loss.
        beta_symm_loss: 0.5
            Scaling factor between symmetry and uniformity loss.
        """
        if self._buse_sim_loss:
            loss = self.calc_sim_loss()
        else:
            loss = self.calc_symm_loss(bsymm_loss, beta_symm_loss)

        return loss

    def calc_sim_loss(self):
        """Calculate MSE deviation from simulation results."""

        scaled_peaks = self._sim_targets * self._center_peak[0]
        loss = ((scaled_peaks - self._peak_pos_map) ** 2).sum()
        loss = loss / self._peak_pos_map.shape[0]

        return loss, loss

    def calc_symm_loss(self, bsymm_loss: bool = True, beta_symm_loss: float = 0.5):
        """Calculate average deviation and loss over map."""

        assert self._peak_pos_map is not None, "ERROR: Map is empty."
        avg_dev = 0.0
        loss = 0.0
        try:
            for peak in self._peak_pos_map:
                avg_dev += (peak - self._center_peak[0]) ** 2
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
                loss_ud = loss_ud / (int(peaks_l.shape[0] * 0.5) * 2.0)

                loss += beta_symm_loss * (loss_ud + loss_lr)
            loss += loss + avg_dev

        except TypeError:
            loss = None
            avg_dev = None

        self.avg_dev = avg_dev
        self.loss = loss

        return avg_dev, loss

    def plot_scanmap(
        self,
        bsavefig: bool = False,
        brel_map: bool = True,
        filename: str = "",
        vlims: list = None,
    ):
        """Make a 2D plot of the scan map results."""

        # det_label = f"Scan {self.label} \n Detector: {self.detector}/1\n"
        det_label = f"Scan {self.label} Detector: {self.detector}/1"

        x = np.unique(self._scan_pos_arr.T[0])
        y = np.unique(self._scan_pos_arr.T[1])
        peak_map = self._peak_pos_map

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
            try:
                if brel_map:
                    data[indy][indx] = f"{peak_map[i]/peak_map[self._mid_ind]:.3f}"
                else:
                    data[indy][indx] = f"{int(peak_map[i])}"
            except TypeError:
                pass

        fig, ax = plt.subplots(figsize=(9, 9))
        if brel_map:
            if vlims is None:
                vlims = [0.996, 1.026]
        else:
            if vlims is None:
                vlims = [None] * 2
        ims = ax.imshow(data, cmap="plasma", vmin=vlims[0], vmax=vlims[1])

        ax.set_xticks(np.arange(x.shape[0]))
        ax.set_yticks(np.arange(y.shape[0]))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)

        # string = "Weights:\n"
        # string = det_label + "\n" + string
        # w_rng = [self.detector * 8, len(self._weights) - 8 * (1 - self.detector)]
        # for i in range(w_rng[0], w_rng[1]):
        #     string += f"pmt{i}/15: {self._weights[i]:.4f}\n"
        # plt.text(0.02, 0.3, string, fontsize=10, transform=plt.gcf().transFigure)
        plt.setp(ax.get_xticklabels(), ha="left")

        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                ax.text(
                    j,
                    i,
                    data[i, j],
                    ha="center",
                    va="center",
                    color="black",
                    bbox={
                        "edgecolor": "white",
                        "facecolor": "white",
                        "alpha": 0.7,
                        "pad": 1.5,
                    },
                )

        try:
            symm_loss = self.loss - self.avg_dev
            # ax.set_title(
            #     f"{det_label} - uniformity term: {self.avg_dev:.0f}, symmetry term: {symm_loss:.0f}"
            # )
        except TypeError:
            pass

        ax.set(xlabel="horizontal position [a.u.]", ylabel="vertical position [a.u.]")
        fig.colorbar(ims)
        fig.tight_layout()

        if bsavefig:
            if filename == "":
                filename = "scan2Dmap"
            plt.savefig(f"{output_path}/{filename}.pdf", dpi=300)

        plt.show()

    @staticmethod
    def _find_closest_ind(all_pos_arr, target_pos):
        """Find index of position closest to target position."""

        assert all_pos_arr.shape[1] == target_pos.shape[0]

        dev = 0.0
        for dim in range(all_pos_arr.shape[1]):
            dev += (all_pos_arr.T[dim] - target_pos[dim]) ** 2
        dev = np.sqrt(dev)

        if dev.min() > 50.0:
            print(
                f"Warning: Could not find target position {target_pos}."
                + f"Min. distance found is {dev.min()}"
            )
        mid_ind = np.argmin(dev)

        return mid_ind

    @staticmethod
    def _get_lr_pairs(pos_arr: np.array):
        """Get left and right positions."""

        x_points = np.unique(pos_arr.T[0])
        x_l = x_points[0]
        x_r = x_points[-1]
        all_left = pos_arr[:, 0] == x_l
        all_right = pos_arr[:, 0] == x_r

        return all_left, all_right


def main():
    pos, evs = scan_200117()
    det = 0
    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        detector=det,
        label=scan_200117.label,
        buse_2Dcorr=True,
        buse_sim_loss=False,
        # mu_init_val=31000.,
        # fit_range=[30000., 32000.],
        do_widths=True,
    )

    smc.calc_peak_positions()
    """
    smc.calc_peak_positions(
        weights=np.array(
            [
                1.008433,
                0.999707,
                0.955058,
                0.98363 ,
                0.99772 ,
                0.989287,
                0.998192,
                0.987019,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
    )
    """

    # {'x_opt': array([1.008433, 0.999707, 0.955058, 0.98363 , 0.99772 , 0.989287, 0.998192, 0.987019]), 'y_opt': (6300.757845958976, 6944.021863459121)}

    print(smc.calc_loss())
    smc.plot_scanmap(bsavefig=False, filename=f"MapOpt{det}")
    # smc.plot_scanmap(brel_map=False)


if __name__ == "__main__":
    main()
