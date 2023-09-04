"""Creating pedestal results over time"""

import datetime
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panter.base.mapPerkeo import MapPerkeo
from panter.config import conf_path
from panter.data.dataMisc import FilePerkeo
from panter.data.dataRootPerkeo import RootPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.pedPerkeo import PedPerkeo
from panter.eval.corrPerkeo import CorrPerkeo
from panter.eval.evalFit import DoFit
from panter.config.evalFitSettings import gaus_simp

output_path = os.getcwd()


class PedMapPerkeo(MapPerkeo):
    """Class for creating and handling of drift correction factors.

    Based on base master class MapPerkeo. Can either create from scratch or import a map
    of pedestal results for each pmt over time.

    Parameters
    ----------
    fmeas: np.array(MeasPerkeo)
        Array of data loader output (DLPerkeo).
    bimp_ped: bool
        To import existing map for final pedestal results.
    outfile_flag: str
        Flag which will be added to output file name. Should be chosen according to
        filter in data loader, i.e. fmeas input.

    Attributes
    ----------
    outfile_flag
    cache: np.array
        Used for storing relevant outputs, besides resulting maps.
        In this case, it is not used.
    maps: list of pd.DataFrame
        map[0]: pedestal map
        Pandas DataFrame with pedestal position and sigma, fit error and rCh2 for each
        PMT with a time stamp. Needs to be imported or calculated.

    Examples
    --------
    Importing existing map with beam flag and plotting result:

    >>> ppm = PedMapPerkeo(filt_meas, bimp_ped=True, outfile_flag="beam")
    >>> ppm()
    >>> ppm.plot_ped_map()

    Starting from scratch:

    >>> file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    >>> dataloader = DLPerkeo(file_dir)
    >>> dataloader.auto()
    >>> filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])
    >>> ppm = PedMapPerkeo(filt_meas, bimp_ped=False, outfile_flag="beam")
    >>> ppm()
    >>> ppm.plot_ped_map()
    """

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_ped: bool = True,
        outfile_flag: str = None,
    ):
        super().__init__(fmeas=fmeas, level=1, bimport=[bimp_ped])
        self.outfile_flag = outfile_flag
        if outfile_flag is None:
            self._outfile = f"ped_map.p"
        else:
            self._outfile = f"ped_map_{outfile_flag}.p"

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """Try to import and/or calculate given level. Return True/False"""

        if level == 0 and bimp:
            # try to import pedestal map
            impfile = FilePerkeo(f"{conf_path}/{self._outfile}")
            self.maps[level], self.cache = impfile.imp()
            assert self.maps[level].shape[0] > 0, "ERROR: Pedestal map empty."

            return True

        elif level == 0 and not bimp:
            # calc ped map
            self.maps[level] = pd.DataFrame(
                columns=[
                    "time",
                    "ped_list",
                    "ped_err_list",
                    "sig_list",
                    "sig_err_list",
                    "rchi2",
                ]
            )
            # self._calc_pedestals()
            # self._calc_pedestals_postcorr()
            self._calc_pedsum_postcorr()

            return True

        else:
            return False

    def _calc_pedestals(self):
        """Calculate pedestals from measurements"""

        for i, meas in enumerate(self._fmeas):
            print(f"Meas No: {i}")
            if i % 100 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile)

            time = meas.date_list[0]

            data = RootPerkeo(meas.file_list[0])
            pedtest = PedPerkeo(
                dataclass=data,
                bplot_res=False,
                bplot_fit=False,
                bplot_log=False,
            )
            ped_list = pedtest.ret_pedestals().T[0]
            ped_err_list = pedtest.ret_pedestals().T[1]
            sig_list = pedtest.ret_pedestals().T[2]
            sig_err_list = pedtest.ret_pedestals().T[3]
            rchi2 = pedtest.ret_pedestals().T[4]

            meas_dict = {
                "time": time,
                "ped_list": ped_list,
                "ped_err_list": ped_err_list,
                "sig_list": sig_list,
                "sig_err_list": sig_err_list,
                "rchi2": rchi2,
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile) == 0
        ), "ERROR: Export of pedestal map failed."

        return 0

    def _calc_pedestals_postcorr(self):
        """Calculate pedestals from measurements after correcting with CorrPerkeo"""

        for i, meas in enumerate(self._fmeas):
            print(f"Meas No: {i} \t {meas}")
            if i % 100 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile)

            time = meas.date_list[0]
            SUM_hist_par = {
                "bin_count": 100,
                "low_lim": -100,
                "up_lim": 100,
            }
            pedcorr_res = [None] * 16
            for det in [0, 1]:
                corr_class = CorrPerkeo(meas, mode=2, custom_pmt_hist_par=SUM_hist_par)
                corr_class.set_all_corr(bactive=False)
                corr_class.corrections["Drift"] = True
                corr_class.corrections["Scan2D"] = False
                corr_class.corrections["RateDepElec"] = True
                corr_class.corrections["Pedestal"] = True
                corr_class.corrections["DeadTime"] = True

                corr_class.addition_filters.append(
                    {
                        "tree": "data",
                        "fkey": "Detector",
                        "active": True,
                        "ftype": "bool",
                        "rightval": 1 - det,
                    }
                )
                corr_class.corr(bstore=True, bwrite=False, bconcat=False)
                histp = corr_class.histograms[0]

                for pmt in range(8 * det, 8 + 8 * det):  # 16
                    test = histp[1][pmt]

                    fitclass = DoFit(test.hist)
                    fitclass.setup(gaus_simp)
                    fitclass.set_bool("boutput", False)
                    fitclass.blogscale = False
                    fitclass.set_fitparam("mu", 0.0)
                    fitclass.set_fitparam("norm", 4000.0)
                    fitclass.limit_range([-1.1 * 28.0, 1.1 * 28.0])
                    fitclass.fit()

                    if fitclass.ret_results() is not None:
                        pedcorr_res[pmt] = [
                            fitclass.ret_results().params["mu"].value,
                            fitclass.ret_results().params["mu"].stderr,
                            np.abs(fitclass.ret_results().params["sig"].value),
                            fitclass.ret_results().params["sig"].stderr,
                            fitclass.ret_gof()[0],
                            fitclass.ret_gof()[1],
                        ]

            pedcorr_res = np.array(pedcorr_res)
            ped_list = pedcorr_res.T[0]
            ped_err_list = pedcorr_res.T[1]
            sig_list = pedcorr_res.T[2]
            sig_err_list = pedcorr_res.T[3]
            rchi2 = pedcorr_res.T[4]

            meas_dict = {
                "time": time,
                "ped_list": ped_list,
                "ped_err_list": ped_err_list,
                "sig_list": sig_list,
                "sig_err_list": sig_err_list,
                "rchi2": rchi2,
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile) == 0
        ), "ERROR: Export of pedestal map failed."

        return 0

    def _calc_pedsum_postcorr(self):
        """Calculate pedestals from measurements after correcting with CorrPerkeo"""

        for i, meas in enumerate(self._fmeas):
            print(f"Meas No: {i} \t {meas}")
            if i % 100 == 0 and i > 0:
                self._write_map2file(map_ind=0, fname=self._outfile)

            time = meas.date_list[0]
            SUM_hist_par = {
                "bin_count": 100,
                "low_lim": -1000,
                "up_lim": 1000,
            }
            pedcorr_res = [None] * 2
            for det in [0, 1]:
                corr_class = CorrPerkeo(meas, mode=1, custom_sum_hist_par=SUM_hist_par)
                corr_class.set_all_corr(bactive=False)
                corr_class.corrections["Drift"] = True
                corr_class.corrections["Scan2D"] = False
                corr_class.corrections["RateDepElec"] = True
                corr_class.corrections["Pedestal"] = True
                corr_class.corrections["DeadTime"] = True

                corr_class.addition_filters.append(
                    {
                        "tree": "data",
                        "fkey": "Detector",
                        "active": True,
                        "ftype": "bool",
                        "rightval": 1 - det,
                    }
                )
                corr_class.corr(bstore=True, bwrite=False, bconcat=False)
                histp = corr_class.histograms[0]

                test = histp[1][det]

                fitclass = DoFit(test.hist)
                fitclass.setup(gaus_simp)
                fitclass.set_bool("boutput", False)
                fitclass.blogscale = False
                fitclass.set_fitparam("mu", 0.0)
                fitclass.set_fitparam("norm", 4000.0)
                fitclass.limit_range([-1.1 * 100.0, 1.1 * 100.0])
                fitclass.fit()

                if fitclass.ret_results() is not None:
                    pedcorr_res[det] = [
                        fitclass.ret_results().params["mu"].value,
                        fitclass.ret_results().params["mu"].stderr,
                        np.abs(fitclass.ret_results().params["sig"].value),
                        fitclass.ret_results().params["sig"].stderr,
                        fitclass.ret_gof()[0],
                        fitclass.ret_gof()[1],
                    ]

            pedcorr_res = np.array(pedcorr_res)
            ped_list = pedcorr_res.T[0]
            ped_err_list = pedcorr_res.T[1]
            sig_list = pedcorr_res.T[2]
            sig_err_list = pedcorr_res.T[3]
            rchi2 = pedcorr_res.T[4]

            meas_dict = {
                "time": time,
                "ped_list": ped_list,
                "ped_err_list": ped_err_list,
                "sig_list": sig_list,
                "sig_err_list": sig_err_list,
                "rchi2": rchi2,
            }
            self.maps[0] = self.maps[0].append(meas_dict, ignore_index=True)

        assert (
            self._write_map2file(map_ind=0, fname=self._outfile) == 0
        ), "ERROR: Export of pedestal map failed."

        return 0

    def plot_ped_map(self, bsave: bool = False):
        """Plot Pedestal map for all PMTs

        Parameters
        ----------
        bsave: False
            Save plot to file. Uses parameter outfile_flag for file name.
        """

        fig, axs = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle(f"Pedestal Map {self.outfile_flag}")

        axs[0].set_title("Pedestals over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [D - M ]", ylabel="Ped pos [ch]")
        axs.flat[1].set(xlabel="Time [D - M ]", ylabel="Ped pos [ch]")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        axs[0].xaxis.set_major_formatter(xfmt)
        axs[1].xaxis.set_major_formatter(xfmt)

        ped_df = self.maps[0]["ped_list"].apply(pd.Series)
        ped_err_df = self.maps[0]["ped_err_list"].apply(pd.Series)

        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

        try:
            for PMT in range(8):
                axs[0].errorbar(
                    dates_plot,
                    ped_df[PMT],
                    yerr=ped_err_df[PMT],
                    fmt=".",
                    label=f"PMT{PMT}",
                )
                axs[1].errorbar(
                    dates_plot,
                    ped_df[PMT + 8],
                    yerr=ped_err_df[PMT + 8],
                    fmt=".",
                    label=f"PMT{PMT + 8}",
                )
        except:
            for PMT in range(2):
                axs[PMT].errorbar(
                    dates_plot,
                    ped_df[PMT],
                    yerr=ped_err_df[PMT],
                    fmt=".",
                    label=f"Det{PMT}",
                )
        axs[0].legend()
        axs[1].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[:-1] + "png", dpi=300)
        plt.show()

        return 0

    def plot_sig_map(self, bsave: bool = False):
        """Plot Pedestal map for all PMTs

        Parameters
        ----------
        bsave: False
            Save plot to file. Uses parameter outfile_flag for file name.
        """

        fig, axs = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle(f"Pedestal Sigma Map {self.outfile_flag}")

        axs[0].set_title("Pedestal Sigmas over time Det 0")
        axs[1].set_title("Sigma over time Det 1")
        axs.flat[0].set(xlabel="Time [D - M ]", ylabel="Sigma [ch]")
        axs.flat[1].set(xlabel="Time [D - M ]", ylabel="Sigma [ch]")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        axs[0].xaxis.set_major_formatter(xfmt)
        axs[1].xaxis.set_major_formatter(xfmt)

        ped_df = self.maps[0]["sig_list"].apply(pd.Series)
        ped_err_df = self.maps[0]["sig_err_list"].apply(pd.Series)

        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]
        try:
            for PMT in range(8):
                axs[0].errorbar(
                    dates_plot,
                    ped_df[PMT],
                    yerr=ped_err_df[PMT],
                    fmt=".",
                    label=f"PMT{PMT}",
                )
                axs[1].errorbar(
                    dates_plot,
                    ped_df[PMT + 8],
                    yerr=ped_err_df[PMT + 8],
                    fmt=".",
                    label=f"PMT{PMT + 8}",
                )
        except:
            for PMT in range(2):
                axs[PMT].errorbar(
                    dates_plot,
                    ped_df[PMT],
                    yerr=ped_err_df[PMT],
                    fmt=".",
                    label=f"Det{PMT}",
                )
        axs[0].legend()
        axs[1].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[:-1] + "png", dpi=300)
        plt.show()

        return 0


# if False:
#     # file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
#     # dataloader = DLPerkeo(file_dir)
#     # dataloader.auto()
#     # filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 4])
#     # filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])
#     sources = np.asarray(["beam", "ce", "cd", "cs", "bi", "sn"])
#     for src in sources:
#         ppm = PedMapPerkeo([], bimp_ped=True, outfile_flag=src)
#         ppm()
#         ppm.plot_ped_map(bsave=True)


def main():
    ind = 5
    file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(file_dir)
    dataloader.auto()
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, ind])[-10::]
    sources = np.asarray(["cd", "ce", "bi", "sn", "cs", "beam"])
    src = sources[ind]
    ppm = PedMapPerkeo(filt_meas, bimp_ped=False, outfile_flag="DetSum" + src)
    ppm()
    ppm.plot_ped_map(bsave=True)
    ppm.plot_sig_map(bsave=True)


if __name__ == "__main__":
    main()
