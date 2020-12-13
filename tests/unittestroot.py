""""""

import os
import subprocess
import numpy as np


class UnitTestRoot:
    """"""

    def __init__(self, test_label: str, params: dict, root_macro: str):
        self.MIN_ACC = 0.00001
        self.rel_dev = None
        self.abs_dev = None
        self.test_label = test_label
        self.params = params
        self.root_macro = root_macro
        self._root_outfile = self.test_label + "_rootout.root"

    def do_root(self, sp_args: list, args: list) -> np.array:
        """"""

        root_cmd = "/home/max/Software/root_install/bin/root"
        this_path = os.path.dirname(os.path.realpath(__file__))
        sp_args.append(self._root_outfile)
        arg_macro = "("
        for sp_arg in sp_args:
            arg_macro += f'"{sp_arg}", '
        for arg in args:
            arg_macro += f"{arg}, "
        arg_macro = arg_macro[:-2]
        arg_macro += ")"

        macro_name = self.root_macro
        arg_old = f"{this_path}/{macro_name}{arg_macro}"
        subprocess.run([root_cmd, arg_old])

        root_outfile = self._root_outfile
        root_res = open(root_outfile).read().split()
        root_res = list(map(float, root_res))
        subprocess.run(["rm", root_outfile])

        return np.array(root_res)

    def do_panter(self) -> np.array:
        """"""

    def check(self, r_res: np.array, p_res: np.array, brel_dev: bool = True):
        """"""

        assert r_res is not None
        assert p_res is not None

        if brel_dev:
            assert (r_res != 0).sum() < 1
            self.rel_dev = (abs(r_res - p_res)).mean() / r_res
            dev = self.rel_dev
        else:
            self.abs_dev = (abs(r_res - p_res)).mean()
            dev = self.abs_dev

        if dev <= self.MIN_ACC:
            print(
                f"GREAT SUCCESS: Unit test passed for test:{self.test_label}. "
                + f"(Abs_Diff is {dev} and needs to be below {self.MIN_ACC})"
            )
            return 0
        else:
            print(
                f"FAILURE: Numbers do not match within precision:"
                + f"{dev} > {self.MIN_ACC}"
            )
            return 1

    def test(self, brel_dev: bool = True, bprint: bool = False, bround: bool = False):
        """"""

        root_res = self.do_root()
        panter_res = self.do_panter()
        if bround:
            root_res = np.round(root_res, 5)
            panter_res = np.round(panter_res, 5)

        if bprint:
            results = np.array([root_res, panter_res]).T
            print(results)
        result = self.check(root_res, panter_res, brel_dev=brel_dev)

        return result
