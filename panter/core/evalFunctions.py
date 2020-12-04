"""Module for storing mathematical functions and distributions."""

import numpy as np
from scipy.special import erfc


def gaussian(x, mu, sig, norm):
    """Standard gaussian distribution."""
    x = np.array(x, dtype=float)
    return (
        norm
        * (1 / (np.sqrt(2 * np.pi * sig ** 2)))
        * np.exp(-((x - mu) ** 2.0) / (2 * sig ** 2.0))
    )


def doublegaussian(x, mu1, sig1, norm1, mu2, sig2, norm2):
    """Sum of two standard gaussian distribution."""
    x = np.array(x, dtype=float)
    return gaussian(x, mu1, sig1, norm1) + gaussian(x, mu2, sig2, norm2)


def f_p0(x, c0):
    """Polynomial of 0th order"""
    x = np.array(x, dtype=float)
    return c0


def f_p1(x, c0, c1):
    """Polynomial of 1st order"""
    x = np.array(x, dtype=float)
    return f_p0(x, c0) + x * c1


def f_p2(x, c0, c1, c2):
    """Polynomial of 2nd order"""
    x = np.array(x, dtype=float)
    return f_p1(x, c0, c1) + c2 * x ** 2


def f_p3(x, c0, c1, c2, c3):
    """Polynomial of 3rd order"""
    x = np.array(x, dtype=float)
    return f_p2(x, c0, c1, c2) + c3 * x ** 3


def fermi_step(x, k, c):
    """Fermi step function."""
    x = np.array(x, dtype=float)
    fermi = 1 / (1 + np.exp(k * (-x + c)))

    return fermi


def exp_dec(x, a, k):
    """Exponential decay function."""
    x = np.array(x, dtype=float)
    return a * np.exp(-x / k)


def exp_sat(x, a, k1, c1):
    """Exponential saturation function."""
    return a - exp_dec(x, c1, k1)


def exp_sat_exp(x, a, k1, c1, k2, c2):
    """Exponential saturation function with 2 indepen. exponentials."""
    return a - exp_dec(x, c1, k1) - exp_dec(x, c2, k2)


def exp_sat_exp2(x, a, k1, c1, k2, c2, k3, c3):
    """Exponential saturation function with 3 indepen. exponentials."""
    return a - exp_dec(x, c1, k1) - exp_dec(x, c2, k2) - exp_dec(x, c3, k3)


def gaussian_pdecay(x, a2, k2, a3, k3, c3):
    """Approx. distribution of individual PMT drift peak."""

    gaus = gaussian(x, c3, k3, a3)
    dec = exp_dec(x, a2, (1.0 / k2))

    return gaus + dec


def appr_erfc(x):
    """Complimentary errforfunction approximation.

    Sergei Winitzki using his 'global Padé approximations'
    """

    a_const = 0.140012
    exp_comp = np.exp(
        -(x ** 2) * (4.0 / np.pi + a_const * x ** 2) / (1 + a_const * x ** 2)
    )

    return np.sign(x) * np.sqrt(1 - exp_comp)


def exmodgaus(x, h, mu, sig, tau):
    """Exponentially modified gaussian distribution."""

    x = np.array(x, dtype=float)
    exp_func_k = h * sig * np.sqrt(np.pi * 0.5) / tau
    exp_func = np.exp(-0.5 * (sig / tau) ** 2 - ((x - mu) / tau))
    err_func = erfc(calc_z(x, mu, sig, tau))

    return exp_func_k * exp_func * err_func


def calc_z(x, mu, sig, tau):
    """Calculate z parameter in exmodgaus for numerical purposes."""

    x = np.array(x, dtype=float)

    return 2 ** (-0.5) * (sig / tau - (x - mu) / sig)


def calc_Acorr_ratedep(
    A_meas: float, A_1: float, t_1: float, delta: float = 0.0033, k: float = 100.0
) -> float:
    """Calculate true ADC value by correcting for rate dependency"""

    A_meas = np.array(A_meas, dtype=float)
    return A_meas + exp_dec(t_1, A_1 * delta, k)
