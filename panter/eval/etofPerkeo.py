"""electron time-of-flight calculation class from data"""

import configparser
import matplotlib.pyplot as plt
import numpy as np

from panter.base.corrSimple import CorrSimple
from panter.config import conf_path
from panter.data.dataMeasPerkeo import MeasPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.data.dataMisc import filt_zeros

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")


class EToFPerkeo:
    """
    Class for generating, plotting and storing eToF histograms of MeasPerkeo objects.
    """

    def __init__(
        self,
        measperkeo: MeasPerkeo,
        bplot_res: bool = False,
        range_detsum: list = None,
        custom_hist_par: dict = None,
    ):
        assert measperkeo.shape[0] == 1, "Class should only be used for single files."
        self._measp = measperkeo
        # self._bplot_log = bplot_log
        self._range_detsum = range_detsum

        if custom_hist_par is None:
            self._etfo_hist_par = {
                "bin_count": int(cnf["dataPerkeo"]["ETOF_hist_counts"]),
                "low_lim": int(cnf["dataPerkeo"]["ETOF_hist_min"]),
                "up_lim": int(cnf["dataPerkeo"]["ETOF_hist_max"]),
            }
        else:
            self._etfo_hist_par = custom_hist_par
        self.tof_hist = None
        self.calc_etof()
        if bplot_res:
            self.plot_etof()

    def calc_etof(self):
        """"""

        corr_class = CorrSimple(
            dataloader=self._measp,
            branch_key="CoinTime",
            hist_par=self._etfo_hist_par,
        )

        # Both detectors must have triggered for eToF
        for det in [0, 1]:
            corr_class.addition_filters.append(
                {
                    "tree": "data",
                    "fkey": "CoinTime",
                    "active": True,
                    "ftype": "num",
                    "low_lim": 0,
                    "up_lim": 4e9,
                    "index": det,
                }
            )
        if self._range_detsum is not None:
            corr_class.addition_filters.append(
                {
                    "tree": "data",
                    "fkey": "DetSum",
                    "active": True,
                    "ftype": "num",
                    "low_lim": self._range_detsum[0],
                    "up_lim": self._range_detsum[1],
                }
            )

        corr_class.corr(bstore=True, bwrite=False)
        histp = corr_class.histograms[0][1][0]
        histp.hist = filt_zeros(histp.hist)
        self.tof_hist = histp

    def plot_etof(self):
        """"""

        self.tof_hist.plot_hist(
            rng=[-25.0, 25.0, 0.0, self.tof_hist.hist["y"].max() * 1.3],
            title=f"eToF - Src{self._measp[0].src} Cyc{self._measp[0].cyc_no}",
            xlabel="eToF [10ns]",
            ylabel="Counts [ ]",
        )


def main():
    dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
    dataloader = DLPerkeo(dir)
    dataloader.auto()

    if 0:
        # beam
        filt_meas = dataloader.ret_filt_meas(["tp", "src", "nomad_no"], [0, 5, 69536])
        # filt_meas[0].tp = 2
    elif 0:
        # cd
        filt_meas = dataloader.ret_filt_meas(
            ["tp", "src", "nomad_no", "cyc_no"], [1, 0, 70378, 194444]
        )[[0]]
        # filt_meas[0].tp = 2
    elif 0:
        # ce
        filt_meas = dataloader.ret_filt_meas(
            ["tp", "src", "nomad_no", "cyc_no"], [1, 1, 70376, 194412]
        )[[0]]
        # filt_meas[0].tp = 2

    elif 0:
        # bi
        filt_meas = dataloader.ret_filt_meas(
            ["tp", "src", "nomad_no", "cyc_no"], [1, 2, 70372, 194348]
        )[[0]]
        filt_meas[0].tp = 2
    elif 1:
        # sn
        filt_meas = dataloader.ret_filt_meas(
            ["tp", "src", "nomad_no", "cyc_no"], [1, 3, 70380, 194464]
        )[[0]]
        # filt_meas[0].tp = 2

    etof = EToFPerkeo(filt_meas, bplot_res=True, range_detsum=[0.0, 7000.0])


if __name__ == "__main__":
    main()
