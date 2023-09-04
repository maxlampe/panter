"""Helper functions for evaluating imported Perkeo root data (RootPerkeo)."""

from __future__ import annotations


import numpy as np


def calc_weights(sample: np.array) -> np.array:
    """Function for calculating weights with 0 exception."""

    weights = []
    for ent in sample:
        assert ent != 0.0, "ERROR: Empty bin in weights calculation."
        weights.append(ent ** (-1))

    return np.array(weights)
