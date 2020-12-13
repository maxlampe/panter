""""""

import panter.core.dataPerkeo as dP

from tests.unittestroot import UnitTestRoot


class HistTestFilter(UnitTestRoot):
    """"""

    def __init__(self, txtfile: str, params: list, root_macro: str):
        super().__init__(
            test_label="HistTestFilter", params=params, root_macro=root_macro
        )
        self.txtfile = txtfile
        self.bfilter = self.params[3]

    def do_panter(self):

        data = dP.RootPerkeo(self.txtfile)
        data.info()
        if self.bfilter:
            data.set_filtdef()
            data.datafilter["DeltaTriggerTime"].active = True
            data.datafilter["DeltaTriggerTime"].upperlimit = 380e3
            data.datafilter["DeltaTriggerTime"].lowerlimit = 650e3
            data.auto(1)
        else:
            data.auto()

        hist_par = {
            "bin_count": self.params[0],
            "low_lim": self.params[1],
            "up_lim": self.params[2],
        }
        data.gen_hist(lpmt=range(data.no_pmts), cust_hist_par=hist_par)
        hpanter1 = data.hists[0]

        return hpanter1.hist.to_numpy().flatten()

    def do_root(self):
        return super().do_root([self.txtfile], self.params)


dir = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
file = dir + "data185836-70086_beam.root"
par1 = [3, 0, 500, 0]
par2 = [3, 0, 500, 1]
root_mac = "filter.cpp"

test_wo_filter = HistTestFilter(txtfile=file, params=par1, root_macro=root_mac)
restult_wo_filter = test_wo_filter.test(brel_dev=False, bprint=True)

test_wi_filter = HistTestFilter(txtfile=file, params=par2, root_macro=root_mac)
restult_wi_filter = test_wi_filter.test(brel_dev=False, bprint=True)

if not restult_wo_filter and not restult_wi_filter:
    print(f"GREAT SUCCESS: Unit test passed with and without filter. ")
else:
    print(
        f"FAILURE: Unit test not passed. Result with and without filter: "
        + f"{restult_wo_filter} / {restult_wi_filter}"
    )
