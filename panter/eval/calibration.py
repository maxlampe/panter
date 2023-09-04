"""WIP. Currently, done manually in panter_applications."""

from panter.base.p3fit_comm import P3FitComm


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
