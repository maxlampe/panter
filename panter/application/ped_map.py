"""Creating pedestal results over time"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
from panter.core.mapPerkeo import MapPerkeo
from panter.core.dataloaderPerkeo import DLPerkeo
from panter.config import conf_path


file_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
dataloader = DLPerkeo(file_dir)
dataloader.auto()
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])
# filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [1, 3, 67732])


class PedMapPerkeo(MapPerkeo):
    """"""

    def __init__(
        self,
        fmeas: np.array = np.asarray([]),
        bimp_ped: bool = True,
        outfile_flag: str = None,
    ):
        super().__init__(fmeas=fmeas, level=1, bimport=[bimp_ped])
        if outfile_flag is None:
            self._outfile = f"ped_map.p"
        else:
            self._outfile = f"ped_map_{outfile_flag}.p"
        self._rch2_limit = 1.5

    def _get_level(self, level: int = 0, bimp: bool = True) -> bool:
        """"""

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

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Pedestal Map")

        axs[0].set_title("Pedestals over time Det 0")
        axs[1].set_title("Peak pos over time Det 1")
        axs.flat[0].set(xlabel="Time [s]", ylabel="Ped pos [ch]")
        axs.flat[1].set(xlabel="Time [s]", ylabel="Ped pos [ch]")

        ped_df = self.maps[0]["ped_list"].apply(pd.Series)
        ped_err_df = self.maps[0]["ped_err_list"].apply(pd.Series)

        for PMT in range(8):
            axs[0].errorbar(
                self.maps[0]["time"],
                ped_df[PMT],
                yerr=ped_err_df[PMT],
                fmt=".",
                label=f"PMT{PMT}",
            )
            axs[1].errorbar(
                self.maps[0]["time"],
                ped_df[PMT + 8],
                yerr=ped_err_df[PMT + 8],
                fmt=".",
                label=f"PMT{PMT + 8}",
            )
        if bsave:
            plt.savefig(self._outfile[:-1] + "png", dpi=300)
        plt.show()

        return 0


ppm = PedMapPerkeo(filt_meas, bimp_ped=False, outfile_flag="beam")
ppm()
ppm.plot_ped_map()
