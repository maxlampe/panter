"""Module for storing mathematical functions and distributions."""

from math import factorial

import numpy as np
from scipy.special import erfc
from scipy.stats import poisson, skewnorm


def gaussian(x, mu, sig, norm: float = 1.0):
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
    return x * 0.0 + c0


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


def exp_dec(x, a, k, x0 = 0.):
    """Exponential decay function."""
    x = np.array(x, dtype=float)
    return a * np.exp(-(x - x0) / k)


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


def pmt_pois(x: int, energy: float, f_pe: float, loc: int = 0):
    """PMT spectra described by a Poisson distribution [Sau18].

    Parameters:
    ----------
    x: int
        Number of photo-electrons generated.
    energy: float
        Electron energy.
    f_pe: float
        Characteristic property of detector. f_pe = a * t * sigma with "a" being the
        Energy to photon conversion, "t" the photon transmission probability and "sigma"
        PMT quantum efficiency.
    loc: int
        Shift x to get non-standardized poisson form.
    """

    x = np.array(x, dtype=int)

    return poisson.pmf(k=x, mu=(f_pe * energy), loc=loc)


def pmt_diffapp(x: float, A: float, B: float):
    """PMT spectra described by a Poisson distribution [Sau18].

    Parameters:
    ----------
    """

    x = np.array(x, dtype=float)

    t1 = np.exp(-A / B)

    pass


# TODO: set h (norm?) to default 1. and change in project acc.
def exmodgaus(x, h, mu, sig, tau):
    """Exponentially modified gaussian distribution."""

    x = np.array(x, dtype=float)
    exp_func_k = h * sig * np.sqrt(np.pi * 0.5) / tau
    exp_func = np.exp(-0.5 * (sig / tau) ** 2 - ((x - mu) / tau))
    err_func = erfc(calc_z(x, mu, sig, tau))

    return exp_func_k * exp_func * err_func


def charge_spec(
    x: float,
    a: float,
    w: float,
    lam: float,
    q0: float,
    sig0: float,
    c0: float,
    sig: float,
    mu: float,
    norm: float = 1.0,
    k_max: int = 100,
):
    """Advanced model for charge spectrum of PMT for absolute calibration

    based on archived paper M. Diwan @ Brookhaven National Labs

    Parameters
    ----------
    x: float
        Charge
    a: float
        ADC to charge
    w: float
        Probability of dark pulse (e.g. thermal electron emission)
    lam: float
        Mean number of photo-electrons at cathode
    q0, sig0: float, float
        Mean and standard deviation of baseline current (raw PMT Pedestal)
    c0: float
        Exponential coefficient for dark pulse component (dark rate parameter)
    mu, sig: float, float
        Mean and standard deviation of gain
    norm: float
        Probability to spectra
    k_max: 10000
        Limit for summation for number of photo-electrons k
    """

    x = a * np.array(x, dtype=float)
    k_max = int(k_max)

    sum = 0.0
    for k in range(k_max):
        mu_k = k * mu + q0
        sig_k = np.sqrt(k * sig ** 2 + sig0 ** 2)
        pois_term = lam ** k * np.exp(-lam) / factorial(k)
        gauss_term = (1 - w) * gaussian(x, mu=mu_k, sig=sig_k)
        emg_term = w * exmodgaus(
            x, h=(np.sqrt(2.0 / np.pi) / sig_k), mu=mu_k, sig=sig_k, tau=c0
        )

        sum += pois_term * gauss_term * emg_term

    return norm * sum


def calc_z(x, mu, sig, tau):
    """Calculate z parameter in exmodgaus for numerical purposes."""

    x = np.array(x, dtype=float)

    return 2 ** (-0.5) * (sig / tau - (x - mu) / sig)


def calc_acorr_ratedep(
    A_meas: float, A_1: float, t_1: float, delta: float = 0.0033, k: float = 100.0
) -> float:
    """Calculate true ADC value by correcting for rate dependency"""

    A_meas = np.array(A_meas, dtype=float)
    return A_meas + exp_dec(t_1, A_1 * delta, k)


def trigger_func(x: float, a: float, p: float):
    """Effective trigger model function according to [Mun06]"""
    return 1.0 - (1.0 - p) ** (a * x) * (1.0 + (a * p * x) / (1.0 - p))


def tof_peaks(
    x: float,
    a: float = 4.0,
    loc: float = 1.0,
    scale: float = 1.0,
    shift: float = 0.0,
    const_bg: float = 0.0,
    norm_p: float = 1.0,
    norm_n: float = 1.0,
):
    """Simple E-ToF model for skewed double peak with relative shift and non-zero bg."""

    x = np.array(x, dtype=float)
    p1 = norm_p * skewnorm.pdf(x, a=a, loc=loc, scale=scale)
    p2 = norm_n * skewnorm.pdf(-(x + shift), a=a, loc=loc, scale=scale)

    return p1 + p2 + const_bg
