"""Dump - For testing and staging"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from panter.core.dataPerkeo import HistPerkeo, RootPerkeo
import panter.core.evalPerkeo as eP
from panter.core.evalFunctions import charge_spec

x = np.linspace(0.0, 10000.0, num=1000)
params = {
    "w": 0.3,
    "a": 0.003,
    "lam": 3.4,
    "q0": 2.0,
    "sig0": 0.9,
    "c0": 10.0,
    "sig": 1.0,
    "mu": 5.0,
    "k_max": 100,
}

val = charge_spec(x, **params)
plt.plot(x, val)
plt.show()
