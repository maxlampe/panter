"""Module for running all unit tests of panter."""

import numpy as np

from tests.histogram import do_histtestbasic
from tests.hist_output import do_histtestout
from tests.filter import do_histtestfilter
from tests.basicfit import do_histtestfit
from tests.background_subtraction import do_backgroundfittest


def run_all():
    """Running all unit tests for a global test of panter."""

    tests = [
        do_histtestbasic(),
        do_histtestout(),
        do_histtestfilter(),
        do_histtestfit(),
        do_backgroundfittest(),
    ]

    print(tests)
    assert np.asarray(tests).sum() == len(tests)

    print("GREAT SUCCESS. \nAll unit tests passed. \nVERY NICE. ")


run_all()
