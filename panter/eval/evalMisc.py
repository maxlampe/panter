"""Module for evaluating imported Perkeo root data (RootPerkeo)."""

from __future__ import annotations

import configparser

import matplotlib.pyplot as plt
import numpy as np

from panter.config import conf_path

# import global analysis parameters
cnf = configparser.ConfigParser()
cnf.read(f"{conf_path}/evalRaw.ini")

# global plot output bool (default is False)
BPLOT = cnf["evalPerkeo"].getboolean("bplot")


def calc_weights(sample: np.array) -> np.array:
    """Function for calculating weights with 0 exception."""

    weights = []
    for ent in sample:
        assert ent != 0.0, "ERROR: Empty bin in weights calculation."
        weights.append(ent ** (-1))

    return np.array(weights)


def scan_fit(param, mod, fres, fdat, bplot, brefit=False):
    # simple routine to take a finished fit, fix all parameters and vary one
    # around its result to return Chi2 and optionally plot it as well

    param_vals = np.linspace(
        fres.params[param].value - 4.0 * fres.params[param].stderr,
        fres.params[param].value + 4.0 * fres.params[param].stderr,
        1000,
    )
    scanres = []
    params = mod.make_params(**fres.params)
    if not brefit:
        params.vary = False
    else:
        params[param].vary = False

    for i in param_vals:
        params[param].value = i
        res = mod.fit(fdat["y"], params, x=fdat["x"], weights=calc_weights(fdat["err"]))
        scanres.append([i, res.chisqr])
    scanres = np.array(scanres)

    if bplot:
        axes = plt.gca()
        axes.set_xlabel(param + " [ ]")
        axes.set_ylabel("Chi^2 [ ]")
        axes.set_xlim([scanres[:, 0].min() * 0.99, scanres[:, 0].max() * 1.02])
        axes.set_ylim(
            [
                2 * scanres[:, 1].min() - scanres[:, 1].max(),
                2 * scanres[:, 1].max() - scanres[:, 1].min(),
            ]
        )
        axes.grid(True)
        plt.plot(scanres[:, 0], scanres[:, 1], ".", label="Scan data")
        plt.axvline(fres.params["norm"].value, -1.0, 1.0, label="Fit result")
        plt.legend(loc="best")
        plt.title("Chi2 scan result")
        plt.tight_layout()
        plt.show()

    return scanres
