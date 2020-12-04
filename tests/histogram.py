""""""

import os
import subprocess
import numpy as np
import panter.core.dataPerkeo as dP

file = "sample.txt"
par = {"HISTP_0": 5, "HISTP_1": 0, "HISTP_2": 15}
root_mac = "histogram.cpp"
root_out = "root_histres.txt"


class UnitTestRoot:
    """"""

    def __init__(self, file_name: str, params: dict, root_macro: str):
        self.MIN_ACC = 0.00001
        self.rel_dev = None
        self.abs_dev = None
        self.test_label = "TEST"
        self.params = params
        self.file = file
        self.root_macro = root_macro

    def do_root(self, macro_name: str, root_outfile: str, *args, **kwargs) -> np.array:
        """"""

        root_cmd = "/home/max/Software/root_install/bin/root"
        this_path = os.path.dirname(os.path.realpath(__file__))
        arg_macro = '('
        for arg in args:
            arg_macro += f'{arg}, '
        arg_macro = arg_macro[:-2]
        arg_macro += ')'

        arg_old = f"{this_path}/{macro_name}{arg_macro}"
        print(arg_old)
        subprocess.run([root_cmd, arg_old])

        root_res = open(root_outfile).read().split()
        root_res = list(map(float, root_res))
        subprocess.run(["rm", root_outfile])

        return np.array(root_res)

    def do_panter(self) -> np.array:
        """"""
        panter_res = np.array([])

        return panter_res

    def check(self, r_res: np.array, p_res: np.array, brel_dev: bool = True):
        """"""

        if brel_dev:
            self.rel_dev = (r_res - p_res).mean() / r_res
            dev = self.rel_dev
        else:
            self.abs_dev = (r_res - p_res).mean()
            dev = self.abs_dev

        if dev <= self.MIN_ACC:
            print(
                f"GREAT SUCCESS: Unit test passed for test:{self.test_label}. "
                + f"(Abs_Diff is {dev} and needs to be below {self.MIN_ACC})"
            )
            return 0
        else:
            print(f"FAILURE: Numbers do not match within precision!")
            return 1

    def test(self):
        """"""

        self.do_root()
        self.do_panter()
        result = self.check(self.get_root(), self.get_panter(), brel_dev=True)

        return result






data_raw = open(file).read().split()
data_raw = list(map(float, data_raw))

hpanter1 = dP.HistPerkeo(data_raw, HISTP_0, HISTP_1, HISTP_2)
hpanter2 = dP.HistPerkeo(np.array(data_raw) + 2, HISTP_0, HISTP_1, HISTP_2)
hpanter1.addhist(hpanter2, -0.5)

print(hpanter1.hist)

abs_dev = ((root_histres - hpanter1.hist.to_numpy().flatten())).mean()


