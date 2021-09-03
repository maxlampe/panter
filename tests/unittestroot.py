"""Unit test base class"""

import os
import subprocess

import numpy as np


class UnitTestRoot:
    """Unit test base class for tests comparing panter results to ROOT results

    Defines necessary functions for this kind of purpose unit test. Made to be
    inherited. All values to be compared have to be equal within a pre-defined accuracy.

    Parameters
    ----------
    test_label : str
        Label of the unit test
    params : list
        List of parameters to be used in unit test
    root_macro : str
        Name of accompanying root macro

    Attributes
    ----------
    MIN_ACC : 0.00001
        Minimum accuracy to be reached by deviation to pass the test
    rel_dev : float
        If calculated, it is the relative deviation from panter and ROOT results.
    abs_dev : float
        If calculated, it is the absolute deviation from panter and ROOT results.
    """

    def __init__(self, test_label: str, params: list, root_macro: str):
        self.MIN_ACC = 0.00001
        self.rel_dev = None
        self.abs_dev = None
        self._test_label = test_label
        self._params = params
        self._root_macro = root_macro
        self._root_outfile = self._test_label + "_rootout.root"

    def _do_root(self, sp_args: list, args: list) -> np.array:
        """Do ROOT evaluation part for the unit test."""

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

        macro_name = self._root_macro
        arg_old = f"{this_path}/{macro_name}{arg_macro}"
        subprocess.run([root_cmd, arg_old])

        root_outfile = self._root_outfile
        root_res = open(root_outfile).read().split()
        root_res = list(map(float, root_res))
        subprocess.run(["rm", root_outfile])

        return np.array(root_res)

    def _do_panter(self) -> np.array:
        """Do panter evaluation part for the unit test."""
        pass

    def _check(self, r_res: np.array, p_res: np.array, brel_dev: bool = True) -> bool:
        """Compare results and check if the mean deviation passes the test."""

        assert r_res is not None
        assert p_res is not None

        if brel_dev:
            assert (r_res != 0).prod() != 0

            self.rel_dev = (abs(r_res - p_res)).mean() / r_res.mean()
            dev = self.rel_dev
        else:
            self.abs_dev = (abs(r_res - p_res)).mean()
            dev = self.abs_dev

        if dev <= self.MIN_ACC:
            print(
                f"GREAT SUCCESS: Unit test passed for test:{self._test_label}. "
                + f"(Abs_Diff is {dev} and needs to be below {self.MIN_ACC})"
            )
            return True
        else:
            print(
                f"FAILURE: Numbers do not match within precision:"
                + f"{dev} > {self.MIN_ACC}"
            )
            print(np.array([r_res, p_res]).T)
            return False

    def test(
        self, brel_dev: bool = True, bprint: bool = False, bround: bool = False
    ) -> bool:
        """Run this unit test."""

        root_res = self._do_root()
        panter_res = self._do_panter()
        if bround:
            root_res = np.round(root_res, 5)
            panter_res = np.round(panter_res, 5)

        if bprint:
            results = np.array([root_res, panter_res]).T
            print(results)
        result = self._check(root_res, panter_res, brel_dev=brel_dev)

        return result
