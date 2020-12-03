import pandas as pd
import numpy as np

import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS

xval = [2.3, 2.9, 3.5, 4.1, 4.7, 5.3, 5.9, 6.5, 7.1, 7.7]
hist = [470, 668, 911, 1107, 1148, 1199, 1067, 928, 691, 506]

data = pd.DataFrame({"x": xval, "y": hist, "err": np.sqrt(hist)})

fitclass = eP.DoFit(data)
fitclass.setup(eFS.gaus_simp)
fitclass.set_fitparam("mu", valpar=5.1)
fitclass.set_fitparam("sig", valpar=2.1)
fitclass.set_fitparam("norm", valpar=200.0)
gausfit = fitclass.fit()
print(gausfit.fit_report())


"""
Root results in default Gaussian fit

   1  Constant     1.19562e+03   1.77219e+01   1.36269e-02  -2.07906e-05
   2  Mean         5.03167e+00   2.91586e-02   3.09479e-05  -1.34615e-02
   3  Sigma        2.00927e+00   3.75556e-02   5.67731e-06   2.09415e-02
Chi2 = 3.73523
ndof = 7
redChi2 = 0.533604
pva = 0.809719
1.7	664	25.7682
2.3	470	21.6795
2.9	668	25.8457
3.5	911	30.1828
4.1	1107	33.2716
4.7	1148	33.8821
5.3	1199	34.6266
5.9	1067	32.665
6.5	928	30.4631
7.1	691	26.2869
7.7	506	22.4944
8.3	641	25.318
8.9	641	25.318
9.5	641	25.318
10.1	641	25.318


Unchanged evalPerkeo result:

    chi-square         = 0.00396078
    reduced chi-square = 5.6583e-04

    mu:    5.03778746 +/- 0.01766753 (0.35%) (init = 5.1)
    sig:   2.00966728 +/- 0.02337377 (1.16%) (init = 2.1)
    norm:  6019.78962 +/- 48.9565229 (0.81%) (init = 200)
    

evalPerkeo without weights:

    chi-square         = 3651.11001
    reduced chi-square = 521.587144

    mu:    5.02482394 +/- 0.02513715 (0.50%) (init = 5.1)
    sig:  -2.01124291 +/- 0.03317915 (1.65%) (init = 2.1)
    norm:  6027.49537 +/- 76.0722837 (1.26%) (init = 200)


Root results with "W" option (i.e. ignoring bin errors)

Constant                  =      1195.59   +/-   12.236      
Mean                      =      5.02482   +/-   0.0251291   
Sigma                     =      2.01124   +/-   0.0327149    	 (limited)
Chi2 = 3651.11
ndof = 7
redChi2 = 521.587
pval = 0

--> evalPerkeo without weights = Root with "W"
    --> weights are the issue

    
evalPerkeo without square for 1/weight

    chi-square         = 3.73522560
    reduced chi-square = 0.53360366

    mu:    5.03167506 +/- 0.02129249 (0.42%) (init = 5.1)
    sig:   2.00926639 +/- 0.02744245 (1.37%) (init = 2.1)
    norm:  6021.72758 +/- 60.1058337 (1.00%) (init = 200)
    
--> equal to root!!!! but why?
"""
