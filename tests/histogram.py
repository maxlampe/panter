""""""

import os
import subprocess
import numpy as np
import panter.core.dataPerkeo as dP

MIN_ACC = 0.00001

HISTP_0 = 5
HISTP_1 = 0
HISTP_2 = 15

file = "sample.txt"
root_cmd = "/home/max/Software/root_install/bin/root"
this_path = os.path.dirname(os.path.realpath(__file__))
arg_old = f"{this_path}/histogram.cpp" + f'("{file}", {HISTP_0}, {HISTP_1}, {HISTP_2})'
subprocess.run([root_cmd, arg_old])

root_histres = open("root_histres.txt").read().split()
root_histres = list(map(float, root_histres))
subprocess.run(["rm", "root_histres.txt"])

data_raw = open(file).read().split()
data_raw = list(map(float, data_raw))

hpanter1 = dP.HistPerkeo(data_raw, HISTP_0, HISTP_1, HISTP_2)
hpanter2 = dP.HistPerkeo(np.array(data_raw) + 2, HISTP_0, HISTP_1, HISTP_2)
hpanter1.addhist(hpanter2, -0.5)

print(hpanter1.hist)

abs_dev = ((root_histres - hpanter1.hist.to_numpy().flatten())).mean()
MIN_ACC
if abs_dev <= MIN_ACC:
    print(
        f"GREAT SUCCESS: Unit test passed for basic histograms with operations. "
        + f"(Abs_Diff needs to be below {MIN_ACC})"
    )
else:
    print(f"FAILURE: Numbers do not match within precision!")
