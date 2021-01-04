"""Module for storing default fit model settings."""

import numpy as np

import panter.core.evalFunctions as eF


class FitSetting:
    """Obj to store/pass fit settings depending on the model used.

    Attributes
    ----------
    label : str
        {'gaus_simp', 'gaus_gen', 'gaus_pdec', 'gaus_expmod',
         'gau_DoGG', 'exp_sat_simp', 'exp_sat_ord2'}
    booldict : dict of bool
        {'boutput': False, 'blimfit': False,
         'bfitall': False,  'brecfit': False}
    initvals : dict
        Initial values for fit for all parameters by keyword.
    paramvary : list of bool
        List of bools whether parameter should be fixed.
    param_limit : dict of [float, float]
        Parameter range fit limits by keyword.
    fitfunc : function
    fitrange : [float, float]
    plotrange : {"x": float, "y": float}
    plot_labels : [str, str, str]
        (PlotLabel, XLabel, YLabel)
    """

    def __init__(self):
        self.label = None
        self.booldict = {
            "boutput": False,
            "blimfit": False,
            "bfitall": False,
            "brecfit": False,
        }
        self.initvals = None
        self.paramvary = None
        self.param_limit = None
        self.fitfunc = None
        self.fitrange = None
        self.plotrange = {"x": None, "y": None}
        self.plot_labels = None


gaus_simp = FitSetting()
gaus_simp.label = "gaus_simp"
gaus_simp.fitfunc = eF.gaussian
gaus_simp.initvals = {"mu": 5000.0, "sig": 300.0, "norm": 1000.0}
gaus_simp.paramvary = [True] * len(gaus_simp.initvals)

gaus_gen = FitSetting()
gaus_gen.label = "gaus_gen"
gaus_gen.fitfunc = eF.gaussian
gaus_gen.initvals = {"mu": 5000.0, "sig": 300.0, "norm": 1000.0}
gaus_gen.paramvary = [True] * len(gaus_gen.initvals)
gaus_gen.booldict["brecfit"] = True

gaus_pdec = FitSetting()
gaus_pdec.label = "gaus_pdec"
gaus_pdec.fitfunc = eF.gaussian_pdecay
gaus_pdec.fitrange = [350.0, 5000.0]
gaus_pdec.booldict["blimfit"] = True
gaus_pdec.initvals = {
    "a2": 200.0,
    "k2": 0.0005,
    "a3": 545232.0,
    "k3": 400.0,
    "c3": 1400.0,
}
gaus_pdec.paramvary = [True] * len(gaus_pdec.initvals)
gaus_pdec.param_limit = {
    "a2": [0.0, np.inf],
    "a3": [0.0, np.inf],
    "k2": [0.0, np.inf],
}

gaus_expmod = FitSetting()
gaus_expmod.label = "gaus_expmod"
gaus_expmod.fitfunc = eF.exmodgaus
gaus_expmod.fitrange = [350.0, 5000.0]
gaus_expmod.booldict["blimfit"] = True
gaus_expmod.initvals = {"h": 80000.0, "mu": 750.0, "sig": 300.0, "tau": 1000.0}
gaus_expmod.paramvary = [True] * len(gaus_expmod.initvals)
gaus_expmod.param_limit = {
    "h": [0.0, np.inf],
    "sig": [0.0001, 10000.0],
    "mu": [0.0, 20000.0],
    "tau": [0.0001, 100000000.0],
}

gaus_DoGG = FitSetting()
gaus_DoGG.label = "gaus_DoGG"
gaus_DoGG.fitfunc = eF.doublegaussian
gaus_DoGG.initvals = {
    "mu1": 5000.0,
    "sig1": 300.0,
    "norm1": 1000.0,
    "mu2": 10000.0,
    "sig2": 300.0,
    "norm2": 1000.0,
}
gaus_DoGG.paramvary = [True] * len(gaus_DoGG.initvals)
gaus_DoGG.param_limit = {"norm1": [0.0, np.inf], "norm2": [0.0, np.inf]}

exp_sat_simp = FitSetting()
exp_sat_simp.label = "exp_sat_simp"
exp_sat_simp.fitfunc = eF.exp_sat
exp_sat_simp.initvals = {"a": 11000.0, "k1": 200.0, "c1": 40.0}
exp_sat_simp.paramvary = [True] * len(exp_sat_simp.initvals)

exp_sat_ord2 = FitSetting()
exp_sat_ord2.label = "exp_sat_ord2"
exp_sat_ord2.fitfunc = eF.exp_sat_exp
exp_sat_ord2.initvals = {
    "a": 11000.0,
    "k1": 200.0,
    "c1": 40.0,
    "k2": 200.0,
    "c2": 40.0,
}
exp_sat_ord2.paramvary = [True] * len(exp_sat_ord2.initvals)
exp_sat_ord2.param_limit = {
    "a": [0.0, 16000.0],
    "k1": [0.001, 10000.0],
    "c1": [-1000.0, 1000.0],
    "k2": [0.001, 10000.0],
    "c2": [-1000.0, 1000.0],
}

pol0 = FitSetting()
pol0.label = "pol0"
pol0.fitfunc = eF.f_p0
pol0.initvals = {"c0": 0.0}
pol0.paramvary = [True] * len(pol0.initvals)

pol1 = FitSetting()
pol1.label = "pol1"
pol1.fitfunc = eF.f_p1
pol1.initvals = {"c0": 0.0, "c1": 1.0}
pol1.paramvary = [True] * len(pol1.initvals)

pol2 = FitSetting()
pol2.label = "pol2"
pol2.fitfunc = eF.f_p2
pol2.initvals = {"c0": 0.0, "c1": 1.0, "c2": 1.0}
pol2.paramvary = [True] * len(pol2.initvals)
