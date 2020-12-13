""""""

import numpy as np

from tests.histogram import do_histtestbasic
from tests.filter import do_histtestfilter
from tests.basicfit import do_histtestfit
from tests.background_subtraction import do_backgroundfittest


def run_all():
    """"""

    tests = [
        do_histtestbasic(),
        do_histtestfilter(),
        do_histtestfit(),
        do_backgroundfittest(),
    ]

    assert np.asarray(tests).sum() == len(tests)

    print("GREAT SUCCESS. \nAll unit tests passed. \nVERY NICE. ")


run_all()
