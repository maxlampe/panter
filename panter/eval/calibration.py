""""""

from panter.base.p3fit_comm import P3FitComm

import os
import numpy as np
from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.corrPerkeo import CorrPerkeo
from panter.eval.evalFit import DoFit
from panter.config.evalFitSettings import gaus_gen

if 0:
    curr_time = None
    for src in range(5):
        data_dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
        dataloader = DLPerkeo(data_dir)
        dataloader.auto()
        if curr_time is None:
            filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, src])[-10]
            curr_time = filt_meas.date_list[0]
            print(curr_time)
        else:
            times = []
            filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, src])
            for meas in filt_meas:
                times.append(meas.date_list[0])
            time_diff = np.abs(np.array(times) - curr_time)
            arg_time = np.argmin(time_diff)

            filt_meas = filt_meas[arg_time]
            print(f"time diff {time_diff[arg_time] / 60.:0.0f} min")

        #     ly_fac = np.array([ly_nonlin_1[src]] * 16)
        #     corr_class = CorrPerkeo(filt_meas, mode=0, weight_arr=ly_fac)
        corr_class = CorrPerkeo(filt_meas, mode=0)
        corr_class.set_all_corr(bactive=False)
        corr_class.corrections["Drift"] = True
        corr_class.corrections["Scan2D"] = True
        corr_class.corrections["RateDepElec"] = True
        corr_class.corrections["Pedestal"] = True
        corr_class.corrections["DeadTime"] = True

        corr_class.addition_filters.append(
            {
                "tree": "data",
                "fkey": "Detector",
                "active": True,
                "ftype": "bool",
                "rightval": 0,
            }
        )

        corr_class.corr(bstore=True, bwrite=True, bconcat=True)

        corr_class.hist_concat.plot_hist()


class CaliPerkeo:
    """
    1) Which sources to fit?
    2) pass fix cycle no?
    3) use closest time or pass all numbers?
    4) create spectra for each detector (write to file)
    5) fit / p3fit
    6) p3fit parameters (fit range, ...); create ini file
    7) store relevant parameters (gain, offset, PE, nonlin k, norm, ...)
    8) p3fit fit results (redchi2, ...) from file
    9) option for iterative fit (start with last gain etc.)
    10) calculate pedestal position and width for each detector
    11) delete spectra files
    12) delete ini file
    13) delete fit results file
    """

    def __init__(self):
        pass

    # def __str__(self):
    #     return "CaliPerkeo"
    #
    # def __repr__(self):
    #     return "CaliPerkeo"

    def __call__(self, *args, **kwargs):
        pass


def main():
    cali = CaliPerkeo()
    print(cali)
    cali()


if __name__ == "__main__":
    main()
