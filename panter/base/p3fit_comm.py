"""Basic communication class for p3fit."""


# https://youtrack.jetbrains.com/issue/PY-29580
import os
import subprocess
import numpy as np


P3FIT_PATH = "/home/max/Software/p3fit_git/p3fit/build/p3fit"
P3FIT_RESULTS_FILE = "fit_res.txt"


class P3FitComm:
    """Basic communication class for p3fit.

    Works by calling p3fit with a script, storing the fit output in a txt file, and
    getting the results by reading the txt file. Very ugly and dirty, but works for
    simple automation.

    Parameters
    ----------
    ini_file, p3fit_path, fit_res_file: str
        Paths to the p3fit ini file, p3fit executable, and name of the output file for
        fit results.

    Attributes
    ----------
    params: dict
        Fitted parameters in a dict.
    gof: dict
        Goodness of fit parameters (rChi2 and p value) in a dict.
    hist_det: dict
        Fitted histograms, e.g., different calibration sources.

    Examples
    --------
    >>> p3_comm = P3FitComm(test_ini)
    >>> p3_comm()
    >>> print(p3_comm.params)
    >>> print(p3_comm.gof)
    """

    def __init__(
        self, ini_file: str = None, p3fit_path: str = None, fit_res_file: str = None
    ):
        self._ini_file = ini_file
        if p3fit_path is not None:
            self._p3fit_path = p3fit_path
        else:
            self._p3fit_path = P3FIT_PATH
        if fit_res_file is not None:
            self._fit_res_file = fit_res_file
        else:
            self._fit_res_file = P3FIT_RESULTS_FILE

        self.params = None
        self.gof = None
        self.hist_det = None

    def __call__(self, *args, **kwargs):
        self._run()
        self._get_results()
        self._remove_results()

    def _run(self):
        """Run p3fit with the ini file."""

        assert self._ini_file is not None, "No ini file set."
        subprocess.run(
            [self._p3fit_path, self._ini_file],
            check=True,
        )

    def _remove_results(self):
        """Remove intermediary output file with fit results."""
        os.remove(self._fit_res_file)

    def _get_results(self):
        """Get fit results from intermediary output file."""

        with open(self._fit_res_file, "r") as file:
            data = file.readlines()[1:]

        end_params_line = None
        tot_rchi_line = None
        start_hist_line = None
        for line_no, line in enumerate(data):
            if line.find("type of") == 0:
                end_params_line = line_no
            elif line.find("red. Chi^2 =") == 0:
                tot_rchi_line = line_no
            elif line.find("histograms:") == 0:
                start_hist_line = line_no + 1

        data = np.array(data)
        params = data[:end_params_line]
        gof = data[tot_rchi_line]
        hist_det = data[start_hist_line:]

        par_dict = {}
        for param in params:
            split = param.split()
            vals = [float(value) if value[0] != "f" else None for value in split[2:]]
            par_dict = {**par_dict, f"{split[1]}": [*vals]}

        gof = {"rChi2": float(gof.split()[5][:-1]), "p-val": float(gof.split()[7])}

        hist_res_dict = {}
        for hist_no, hist in enumerate(hist_det):
            split = hist.split()
            hist_res_dict = {**hist_res_dict, f"{hist_no}": [hist, float(split[-1])]}

        self.params = par_dict
        self.gof = gof
        self.hist_det = hist_res_dict


def main():
    os.chdir("/home/max/Software/panter_applications/analyse_max/auto_fierz")
    test_ini = "/home/max/Software/panter_applications/analyse_max/auto_fierz/p3fit_calibration.ini"
    p3_comm = P3FitComm(test_ini)
    p3_comm()

    print(p3_comm.params)
    print(p3_comm.gof)
    print(p3_comm.hist_det)


if __name__ == "__main__":
    main()
