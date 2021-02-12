"""Dump"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panter.core.dataPerkeo as dP
import panter.core.evalPerkeo as eP
import panter.config.evalFitSettings as eFS
from panter.core.dataloaderPerkeo import DLPerkeo, MeasPerkeo
from panter.core.corrPerkeo import corrPerkeo

from panter import output_path

print(output_path)

exit()

test = [1, 2, 3]
file = dP.FilePerkeo("test.p")
file.dump(test)
