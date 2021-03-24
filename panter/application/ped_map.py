"""Creating pedestal results over time"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
from panter.core.mapPerkeo import MapPerkeo
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.config import conf_path
from panter import output_path


class PedMapPerkeo(MapPerkeo):
    """Class for creating and handling of drift correction factors.

    Based on core master class MapPerkeo. Can either create from scratch or import a map
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
            impfile = dP.FilePerkeo(f"{conf_path}/{self._outfile}")
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
            self._calc_pedestals()

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

            data = dP.RootPerkeo(meas.file_list[0])
            pedtest = eP.PedPerkeo(
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

    def plot_ped_map(self, bsave: bool = False):
        """Plot Pedestal map for all PMTs"""

        fig, axs = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle(f"Pedestal Map {self.outfile_flag}")

        axs[0].set_title("Pedestals over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [D - M ]", ylabel="Ped pos [ch]")
        axs.flat[1].set(xlabel="Time [ ]", ylabel="Ped pos [ch]")

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        axs[0].xaxis.set_major_formatter(xfmt)
        axs[1].xaxis.set_major_formatter(xfmt)

        ped_df = self.maps[0]["ped_list"].apply(pd.Series)
        ped_err_df = self.maps[0]["ped_err_list"].apply(pd.Series)

        dates_plot = [datetime.datetime.fromtimestamp(t) for t in self.maps[0]["time"]]

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
        axs[0].legend()
        axs[1].legend()
        if bsave:
            plt.savefig(output_path + "/" + self._outfile[:-1] + "png", dpi=300)
        plt.show()

        return 0


if False:
    # file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    # dataloader = DLPerkeo(file_dir)
    # dataloader.auto()
    # filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 4])
    # filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])
    sources = np.asarray(["beam", "ce", "cd", "cs", "bi", "sn"])
    for src in sources:
        ppm = PedMapPerkeo([], bimp_ped=True, outfile_flag=src)
        ppm()
        ppm.plot_ped_map(bsave=True)


if False:
    ind = 1
    file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(file_dir)
    dataloader.auto()
    filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, ind])
    sources = np.asarray(["cd", "ce", "bi", "sn", "cs", "beam"])
    src = sources[ind]
    ppm = PedMapPerkeo(filt_meas, bimp_ped=False, outfile_flag=src)
    ppm()
    ppm.plot_ped_map(bsave=True)
