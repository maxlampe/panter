"""Calculate drift map from Sn measurements."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
from panter.core.mapPerkeo import MapPerkeo
from panter.config.evalFitSettings import gaus_expmod
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo
from panter.config import conf_path
from panter import output_path

file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(file_dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 3])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])


class DriftMapPerkeo(MapPerkeo):
    """Class for creating and handling of drift correction factors.

    Based on core master class MapPerkeo. Can either create from scratch a map of Sn
    drift measurement fits to single PMT spectra or use the latter and create a drift
    correction factor map for each PMT. The maps can either be imported or created anew.
    For the drift correction factors, the relative deviation to the weighted arithmetic
    mean of the individual PMT Sn peak position is used. Sn fits with a redChi2 above
    1.5 are ignored.

    Parameters
    ----------
    fmeas: np.array(MeasPerkeo)
        Array of data loader output (DLPerkeo).
    bimp_pmt, bimp_sn : bool
        To import existing maps for final pmt correction factors or Sn peak positions
        respectively.

    Attributes
    ----------
    cache: np.array
        Used for storing relevant outputs, besides resulting maps. In this case, an
        array of weighted arithmetic means of Sn peak positions for each PMT channel.
        Needs to be imported or calculated with Sn map (sn_map).
    map: list of pd.DataFrame
        map[0]: pmt factor map
        Pandas DataFrame with drift correction factors for each PMT with a time stamp.
        Needs to be imported or calculated from Sn map (sn_map).
        map[1]: sn peak pos map
        Pandas DataFrame with drift Sn peak positions for each PMT with a time stamp.
        Needs to be imported or calculated from data loader files.

    Examples
    --------
    Importing existing map and plotting result:

    >>> pdm = DriftMapPerkeo()
    >>> pdm()
    >>> pdm.plot_pmt_map()

    Starting from scratch:

    >>> file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    >>> dataloader = DLPerkeo(file_dir)
    >>> dataloader.auto()
    >>> filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 3])
    >>> pdm = DriftMapPerkeo(fmeas=filt_meas, bimp_pmt=False, bimp_sn=False)
    >>> pdm()
    >>> pdm.plot_sn_map()
    >>> pdm.plot_pmt_map()
    """

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_pmt: bool = True,
        bimp_sn: bool = False,
    ):
        super().__init__(fmeas=fmeas, level=2, bimport=[bimp_pmt, bimp_sn])
        self._outfile = ["sn_peak_map.p", "pmt_fac_map.p"]
        self._rch2_limit = 1.5

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""

        if level == 0 and bimp:
            # try to import pmt factor map
            impfile = dP.FilePerkeo(f"{conf_path}/{self._outfile[1]}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[level].shape[0] > 0, "ERROR: PMT factor map empty."

            return True

        elif level == 1 and bimp:
            # try to import sn peak map
            impfile = dP.FilePerkeo(f"{conf_path}/{self._outfile[0]}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[1].shape[0] > 0

            self.maps[level - 1] = pd.DataFrame(columns=["time", "pmt_fac"])
            self._calc_pmt_fac()

            return True

        elif level == 1 and not bimp:
            self.maps[level] = pd.DataFrame(
                columns=["time", "peak_list", "err_list", "rchi2"]
            )
            self._calc_peak_pos()
            self.maps[level - 1] = pd.DataFrame(columns=["time", "pmt_fac"])
            self._calc_pmt_fac()

            return True

        else:
            return False

    def _calc_peak_pos(self):
        """Calculate peak position from Sn drift measurements"""

        for i, meas in enumerate(self._fmeas):
            if i in [61, 109, 294, 411, 625]:
                continue

            print(f"Meas No: {i}")
            if i % 100 == 0 and i > 0:
                self._write_map2file(map_ind=1, fname=self._outfile[0])

            time = meas.date_list[0]
            # if time > 2.5787e9:
            #    continue

            corr_class = CorrPerkeo(dataloader=meas, mode=2)
            corr_class.set_all_corr(bactive=False)
            corr_class.corrections["DeadTime"] = True
            corr_class.corrections["Pedestal"] = True
            corr_class.corrections["RateDepElec"] = True

            corr_class.corr(bstore=True, bwrite=False)

            fitsettings = gaus_expmod
            fitsettings.plot_labels = [
                "SnSpec fit result",
                "ADC [ch]",
                "Counts [ ]",
            ]
            fitsettings.plotrange["x"] = [30.0, 4000.0]

            hists = corr_class.histograms[0][1]
            rchi2 = np.array([])
            mu_val = np.array([])
            mu_err = np.array([])

            for j in range(len(hists)):
                dofitclass = eP.DoFit(hists[j].hist)
                dofitclass.setup(fitsettings)
                dofitclass.set_bool("boutput", False)
                dofitclass.fit()

                fit_results = dofitclass.ret_results()
                rchi2 = np.append(rchi2, dofitclass.ret_gof()[0])
                mu_val = np.append(mu_val, fit_results.params["mu"].value)
                mu_err = np.append(mu_err, fit_results.params["mu"].stderr)

            meas_dict = {
                "time": time,
                "peak_list": mu_val,
                "err_list": mu_err,
                "rchi2": rchi2,
            }
            self.maps[1] = self.maps[1].append(meas_dict, ignore_index=True)

        rchi2_df = self.maps[1]["rchi2"].apply(pd.Series)
        peak_df = self.maps[1]["peak_list"].apply(pd.Series)
        err_df = self.maps[1]["err_list"].apply(pd.Series)

        rchi2_filter = rchi2_df < self._rch2_limit
        peak_df = peak_df[rchi2_filter]
        err_df = err_df[rchi2_filter]

        self.cache = (peak_df / err_df ** 2).sum() / (1.0 / err_df ** 2).sum()

        assert (
            self._write_map2file(map_ind=1, fname=self._outfile[0]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def _calc_pmt_fac(self):
        """Calculate drift correction factors for each PMT from sn map"""

        for index, sn_meas in self.maps[1].iterrows():

            factors = (self.cache / sn_meas["peak_list"]).to_numpy()
            print(factors)
            rchi2_filter = sn_meas["rchi2"] > self._rch2_limit
            factors[rchi2_filter] = None
            print(factors)
            pmt_dict = {
                "time": sn_meas["time"],
                "pmt_fac": factors,
            }

            self.maps[0] = self.maps[0].append(pmt_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile[1]) == 0
        ), "ERROR: Export of drift map failed."

        return 0

    def plot_sn_map(self, bsave: bool = False):
        """Plot Sn drift map for all PMTs"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Sn Drift study")

        axs[0].set_title("Peak pos over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="EMG peak pos [ch]")

        peak_df = self.maps[1]["peak_list"].apply(pd.Series)
        err_df = self.maps[1]["err_list"].apply(pd.Series)

        for PMT in range(8):
            axs[0].errorbar(
                self.maps[1]["time"],
                peak_df[PMT],
                yerr=err_df[PMT],
                fmt=".",
                label=f"PMT{PMT}",
            )
            axs[1].errorbar(
                self.maps[1]["time"],
                peak_df[PMT + 8],
                yerr=err_df[PMT + 8],
                fmt=".",
                label=f"PMT{PMT + 8}",
            )
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[0][:-1] + "png", dpi=300)
        plt.show()

        return 0

    def plot_pmt_map(self, bsave: bool = False):
        """Plot drift correction factor map for all PMTs"""

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("PMT drift correction factors")

        axs[0].set_title("Factor over time Det 0")
        axs[1].set_title("Factor over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="Correction factor [ ]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="Correction factor [ ]")

        pmt_fac = self.maps[0]["pmt_fac"].apply(pd.Series)

        for PMT in range(8):
            axs[0].plot(
                self.maps[0]["time"],
                pmt_fac[PMT],
                ".",
                label=f"PMT{PMT}",
            )
            axs[1].plot(
                self.maps[0]["time"],
                pmt_fac[PMT + 8],
                ".",
                label=f"PMT{PMT + 8}",
            )
        axs[0].legend()
        axs[1].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[1][:-1] + "png", dpi=300)
        plt.show()

        return 0


pdm = DriftMapPerkeo(filt_meas)
pdm()
# pdm.plot_sn_map()
pdm.plot_pmt_map()
