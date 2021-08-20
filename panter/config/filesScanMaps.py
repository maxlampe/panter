"""header for scanfiles"""

import numpy as np

DEF_SIG_DIR = "/mnt/sda/directbackup_perkeo33/dataperkeo19/scan"
DEF_BG_DIR = "/mnt/sda/directbackup_perkeo33/dataperkeo19/cycle201"


class ScanFiles:
    """"""

    def __init__(
        self,
        dir_sig_files: str,
        dir_bg_files: str = None,
        sig_files: list = None,
        bg_files: list = None,
        positions: list = None,
        cyc_numbers: list = None,
        label: str = "unlabelled",
        measp_type: int = 1,
        measp_src: int = 3,
    ):
        self._dir_sig_files = dir_sig_files
        self._dir_bg_files = dir_bg_files
        self._label = label
        self._measp_type = measp_type
        self._measp_src = measp_src

        self._sig_files = sig_files
        self._bg_files = bg_files
        self._positions = positions
        self._cyc_numbers = cyc_numbers
        self.events = None

    def __call__(self):
        if self._dir_sig_files[-1:] != "/":
            self._dir_sig_files = self._dir_sig_files + "/"
        if self._dir_bg_files is None:
            self._dir_bg_files = self._dir_sig_files
        if self._dir_bg_files[-1:] != "/":
            self._dir_bg_files = self._dir_bg_files + "/"

        ev_list = []
        for ind, sig_file in enumerate(self._sig_files):
            curr_ev = [self._measp_type, self._measp_src]
            curr_sig_file = self._dir_sig_files + sig_file
            curr_bg_file = self._dir_bg_files + self._bg_files[ind]
            curr_ev.append(list([curr_sig_file, curr_bg_file]))

            if self._cyc_numbers is not None:
                curr_ev.append(self._cyc_numbers[ind])
            ev_list.append(curr_ev)

        self.events = ev_list

        return self._positions, self.events

    def __str__(self):
        return "ScanFiles" + "_" + self._label

    def __repr__(self):
        return "ScanFiles" + "_" + self._label


pos = np.asarray(
    [
        [20, 8776],
        [20, 6772],
        [20, 4768],
        [20, 2764],
        [170, 2764],
        [170, 4768],
        [170, 6772],
        [170, 8776],
        [320, 8776],
        [320, 6772],
        [320, 4768],
        [320, 2764],
    ]
)
evs = np.asarray(
    [
        ["data128986--1_20_8776_3.root", "data128784-67916_bg.root", 128986],
        ["data128992--1_20_6772_3.root", "data128784-67916_bg.root", 128992],
        ["data128998--1_20_4768_3.root", "data128784-67916_bg.root", 128998],
        ["data129004--1_20_2764_3.root", "data128784-67916_bg.root", 129004],
        ["data129010--1_170_2764_3.root", "data128784-67916_bg.root", 129010],
        ["data129016--1_170_4768_3.root", "data128784-67916_bg.root", 129016],
        ["data129022--1_170_6772_3.root", "data129070-67926_bg.root", 129022],
        ["data129028--1_170_8776_3.root", "data129070-67926_bg.root", 129028],
        ["data129034--1_320_8776_3.root", "data129070-67926_bg.root", 129034],
        ["data129040--1_320_6772_3.root", "data129070-67926_bg.root", 129040],
        ["data129046--1_320_4768_3.root", "data129070-67926_bg.root", 129046],
        ["data129052--1_320_2764_3.root", "data129070-67926_bg.root", 129052],
    ]
)
scan_200112 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200112",
)


pos = np.asarray(
    [
        [20, 2764],
        [20, 3766],
        [20, 4768],
        [20, 5770],
        [20, 6772],
        [20, 7774],
        [20, 8776],
        [95, 2764],
        [95, 3766],
        [95, 4768],
        [95, 5770],
        [95, 6772],
        [95, 7774],
        [95, 8776],
        [170, 2764],
        [170, 3766],
        [170, 4768],
        [170, 5770],
        [170, 6772],
        [170, 7774],
        [170, 8776],
        # [245, 2764], measurement repetition
        [245, 2764],
        [245, 3766],
        [245, 4768],
        [245, 5770],
        [245, 6772],
        [245, 7774],
        [245, 8776],
        [320, 2764],
        [320, 3766],
        [320, 4768],
        [320, 5770],
        [320, 6772],
        [320, 7774],
        [320, 8776],
    ]
)
evs = np.asarray(
    [
        ["data160774--1_20_2764_3.root", "data160770--1_bg.root", 160774],
        ["data160780--1_20_3766_3.root", "data160770--1_bg.root", 160780],
        ["data160786--1_20_4768_3.root", "data160770--1_bg.root", 160786],
        ["data160792--1_20_5770_3.root", "data160770--1_bg.root", 160792],
        ["data160798--1_20_6772_3.root", "data160816--1_bg.root", 160798],
        ["data160804--1_20_7774_3.root", "data160816--1_bg.root", 160804],
        ["data160810--1_20_8776_3.root", "data160816--1_bg.root", 160810],
        ["data160820--1_95_2764_3.root", "data160816--1_bg.root", 160820],
        ["data160826--1_95_3766_3.root", "data160816--1_bg.root", 160826],
        ["data160832--1_95_4768_3.root", "data160816--1_bg.root", 160832],
        ["data160838--1_95_5770_3.root", "data160816--1_bg.root", 160838],
        ["data160844--1_95_6772_3.root", "data160862--1_bg.root", 160844],
        ["data160850--1_95_7774_3.root", "data160862--1_bg.root", 160850],
        ["data160856--1_95_8776_3.root", "data160862--1_bg.root", 160856],
        ["data160866--1_170_2764_3.root", "data160862--1_bg.root", 160866],
        ["data160872--1_170_3766_3.root", "data160862--1_bg.root", 160872],
        ["data160878--1_170_4768_3.root", "data160862--1_bg.root", 160878],
        ["data160884--1_170_5770_3.root", "data160862--1_bg.root", 160884],
        ["data160890--1_170_6772_3.root", "data160908--1_bg.root", 160890],
        ["data160896--1_170_7774_3.root", "data160908--1_bg.root", 160896],
        ["data160902--1_170_8776_3.root", "data160908--1_bg.root", 160902],
        # ["data160912--1_245_2764_3.root", "data160908--1_bg.root", 160912],
        ["data160934--1_245_2764_3.root", "data160930--1_bg.root", 160934],
        ["data160940--1_245_3766_3.root", "data160930--1_bg.root", 160940],
        ["data160946--1_245_4768_3.root", "data160930--1_bg.root", 160946],
        ["data160952--1_245_5770_3.root", "data160930--1_bg.root", 160952],
        ["data160958--1_245_6772_3.root", "data160976--1_bg.root", 160958],
        ["data160964--1_245_7774_3.root", "data160976--1_bg.root", 160964],
        ["data160970--1_245_8776_3.root", "data160976--1_bg.root", 160970],
        ["data160980--1_320_2764_3.root", "data160976--1_bg.root", 160980],
        ["data160986--1_320_3766_3.root", "data160976--1_bg.root", 160986],
        ["data160992--1_320_4768_3.root", "data160976--1_bg.root", 160992],
        ["data160998--1_320_5770_3.root", "data161022--1_bg.root", 160998],
        ["data161004--1_320_6772_3.root", "data161022--1_bg.root", 161004],
        ["data161010--1_320_7774_3.root", "data161022--1_bg.root", 161010],
        ["data161016--1_320_8776_3.root", "data161022--1_bg.root", 161016],
    ]
)
scan_200117 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200117",
)


pos = np.asarray(
    [
        [20, 8776],
        [20, 7774],
        [20, 6772],
        [20, 5770],
        [20, 4768],
        [20, 3766],
        [20, 2764],
        [95, 2764],
        [95, 3766],
        [95, 4768],
        [95, 5770],
        [95, 6772],
        [95, 7774],
        [95, 8776],
        [170, 8776],
        [170, 7774],
        [170, 6772],
        [170, 5770],
        [170, 4768],
        [170, 3766],
        [170, 2764],
        [245, 2764],
        [245, 3766],
        [245, 4768],
        [245, 5770],
        [245, 6772],
        [245, 7774],
        [245, 8776],
        [320, 8776],
        [320, 7774],
        [320, 6772],
        [320, 5770],
        [320, 4768],
        [320, 3766],
        [320, 2764],
    ]
)
evs = np.asarray(
    [
        ["data135346--1_20_8776_3.root", "data135556.root", 135346],
        ["data135352--1_20_7774_3.root", "data135556.root", 135352],
        ["data135358--1_20_6772_3.root", "data135556.root", 135358],
        ["data135364--1_20_5770_3.root", "data135556.root", 135364],
        ["data135370--1_20_4768_3.root", "data135556.root", 135370],
        ["data135376--1_20_3766_3.root", "data135556.root", 135376],
        ["data135382--1_20_2764_3.root", "data135556.root", 135382],
        ["data135388--1_95_2764_3.root", "data135556.root", 135388],
        ["data135394--1_95_3766_3.root", "data135556.root", 135394],
        ["data135400--1_95_4768_3.root", "data135556.root", 135400],
        ["data135406--1_95_5770_3.root", "data135556.root", 135406],
        ["data135412--1_95_6772_3.root", "data135556.root", 135412],
        ["data135418--1_95_7774_3.root", "data135556.root", 135418],
        ["data135424--1_95_8776_3.root", "data135556.root", 135424],
        ["data135430--1_170_8776_3.root", "data135556.root", 135430],
        ["data135436--1_170_7774_3.root", "data135556.root", 135436],
        ["data135442--1_170_6772_3.root", "data135556.root", 135442],
        ["data135448--1_170_5770_3.root", "data135556.root", 135448],
        ["data135454--1_170_4768_3.root", "data135556.root", 135454],
        ["data135460--1_170_3766_3.root", "data135556.root", 135460],
        ["data135466--1_170_2764_3.root", "data135556.root", 135466],
        ["data135472--1_245_2764_3.root", "data135556.root", 135472],
        ["data135478--1_245_3766_3.root", "data135556.root", 135478],
        ["data135484--1_245_4768_3.root", "data135556.root", 135484],
        [
            "data135490--1_245_5770_3.root",
            "data135556.root",
            135490,
        ],  # bad hist detector1
        [
            "data135496--1_245_6772_3.root",
            "data135556.root",
            135496,
        ],  # bad hist detector1
        ["data135502--1_245_7774_3.root", "data135556.root", 135502],
        ["data135508--1_245_8776_3.root", "data135556.root", 135508],
        ["data135514--1_320_8776_3.root", "data135556.root", 135514],
        [
            "data135520--1_320_7774_3.root",
            "data135556.root",
            135520,
        ],  # bad hist detector1
        [
            "data135526--1_320_6772_3.root",
            "data135556.root",
            135526,
        ],  # bad hist detector1
        ["data135532--1_320_5770_3.root", "data135556.root", 135532],
        ["data135538--1_320_4768_3.root", "data135556.root", 135538],
        ["data135544--1_320_3766_3.root", "data135556.root", 135544],
        ["data135550--1_320_2764_3.root", "data135556.root", 135550],
    ]
)
scan_200113 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200113",
)


pos = np.asarray(
    [
        [20, 8776],
        [20, 7774],
        [20, 6772],
        [20, 5770],
        [20, 4768],
        [20, 3766],
        [20, 2764],
        [95, 2764],
        [95, 3766],
        [95, 4768],
        [95, 5770],
        [95, 6772],
        [95, 7774],
        [95, 8776],
        [170, 8776],
        [170, 7774],
        [170, 6772],
        [170, 5770],
        [170, 4768],
        [170, 3766],
        [170, 2764],
        [245, 2764],
        [245, 3766],
        [245, 4768],
        [245, 5770],
        [245, 6772],
        [245, 7774],
        [245, 8776],
        [320, 8776],
        [320, 7774],
        [320, 6772],
        [320, 5770],
        [320, 4768],
        [320, 3766],
        [320, 2764],
    ]
)
evs = np.asarray(
    [
        ["data141865--1_20_8776_3.root", "data141855-68496_bg.root", 141865],
        ["data141871--1_20_7774_3.root", "data141855-68496_bg.root", 141871],
        ["data141877--1_20_6772_3.root", "data141855-68496_bg.root", 141877],
        ["data141883--1_20_5770_3.root", "data141855-68496_bg.root", 141883],
        ["data141889--1_20_4768_3.root", "data141855-68496_bg.root", 141889],
        ["data141895--1_20_3766_3.root", "data141855-68496_bg.root", 141895],
        ["data141901--1_20_2764_3.root", "data141855-68496_bg.root", 141901],
        ["data141907--1_95_2764_3.root", "data141855-68496_bg.root", 141907],
        ["data141913--1_95_3766_3.root", "data141855-68496_bg.root", 141913],
        ["data141919--1_95_4768_3.root", "data141855-68496_bg.root", 141919],
        ["data141925--1_95_5770_3.root", "data141855-68496_bg.root", 141925],
        ["data141931--1_95_6772_3.root", "data141855-68496_bg.root", 141931],
        ["data141937--1_95_7774_3.root", "data141855-68496_bg.root", 141937],
        ["data141943--1_95_8776_3.root", "data141855-68496_bg.root", 141943],
        ["data141949--1_170_8776_3.root", "data141855-68496_bg.root", 141949],
        ["data141955--1_170_7774_3.root", "data141855-68496_bg.root", 141955],
        ["data141961--1_170_6772_3.root", "data141855-68496_bg.root", 141961],
        ["data141967--1_170_5770_3.root", "data142093-68498_bg.root", 141967],
        ["data141973--1_170_4768_3.root", "data142093-68498_bg.root", 141973],
        ["data141979--1_170_3766_3.root", "data142093-68498_bg.root", 141979],
        ["data141985--1_170_2764_3.root", "data142093-68498_bg.root", 141985],
        ["data141991--1_245_2764_3.root", "data142093-68498_bg.root", 141991],
        ["data141997--1_245_3766_3.root", "data142093-68498_bg.root", 141997],
        ["data142003--1_245_4768_3.root", "data142093-68498_bg.root", 142003],
        ["data142009--1_245_5770_3.root", "data142093-68498_bg.root", 142009],
        ["data142015--1_245_6772_3.root", "data142093-68498_bg.root", 142015],
        ["data142021--1_245_7774_3.root", "data142093-68498_bg.root", 142021],
        ["data142027--1_245_8776_3.root", "data142093-68498_bg.root", 142027],
        ["data142033--1_320_8776_3.root", "data142093-68498_bg.root", 142033],
        ["data142039--1_320_7774_3.root", "data142093-68498_bg.root", 142039],
        ["data142045--1_320_6772_3.root", "data142093-68498_bg.root", 142045],
        ["data142051--1_320_5770_3.root", "data142093-68498_bg.root", 142051],
        ["data142057--1_320_4768_3.root", "data142093-68498_bg.root", 142057],
        ["data142063--1_320_3766_3.root", "data142093-68498_bg.root", 142063],
        ["data142069--1_320_2764_3.root", "data142093-68498_bg.root", 142069],
    ]
)
scan_200114 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200114",
)


pos = np.asarray(
    [
        [20, 2764],
        [20, 3766],
        [20, 4768],
        [20, 5770],
        [20, 6772],
        [20, 7774],
        [20, 8776],
        [95, 2764],
        [95, 3766],
        [95, 4768],
        [95, 5770],
        [95, 6772],
        [95, 7774],
        [95, 8776],
        [170, 2764],
        [170, 3766],
        [170, 4768],
        [170, 5770],
        [170, 6772],
        [170, 7774],
        [170, 8776],
        [245, 2764],
        [245, 3766],
        [245, 4768],
        [245, 5770],
        [245, 6772],
        [245, 7774],
        [245, 8776],
        [320, 2764],
        [320, 3766],
        [320, 4768],
        [320, 5770],
        [320, 6772],
        [320, 7774],
        [320, 8776],
    ]
)
evs = np.asarray(
    [
        ["data176706--1_20_2764_3.root", "data176702--1_bg.root", 176706],
        ["data176712--1_20_3766_3.root", "data176702--1_bg.root", 176712],
        ["data176718--1_20_4768_3.root", "data176702--1_bg.root", 176718],
        ["data176724--1_20_5770_3.root", "data176702--1_bg.root", 176724],
        ["data176730--1_20_6772_3.root", "data176748--1_bg.root", 176730],
        ["data176736--1_20_7774_3.root", "data176748--1_bg.root", 176736],
        ["data176742--1_20_8776_3.root", "data176748--1_bg.root", 176742],
        ["data176752--1_95_2764_3.root", "data176748--1_bg.root", 176752],
        ["data176758--1_95_3766_3.root", "data176748--1_bg.root", 176758],
        ["data176764--1_95_4768_3.root", "data176748--1_bg.root", 176764],
        ["data176770--1_95_5770_3.root", "data176748--1_bg.root", 176770],
        ["data176776--1_95_6772_3.root", "data176794--1_bg.root", 176776],
        ["data176782--1_95_7774_3.root", "data176794--1_bg.root", 176782],
        ["data176788--1_95_8776_3.root", "data176794--1_bg.root", 176788],
        ["data176798--1_170_2764_3.root", "data176794--1_bg.root", 176798],
        ["data176804--1_170_3766_3.root", "data176794--1_bg.root", 176804],
        ["data176810--1_170_4768_3.root", "data176794--1_bg.root", 176810],
        ["data176816--1_170_5770_3.root", "data176794--1_bg.root", 176816],
        ["data176822--1_170_6772_3.root", "data176840--1_bg.root", 176822],
        ["data176828--1_170_7774_3.root", "data176840--1_bg.root", 176828],
        ["data176834--1_170_8776_3.root", "data176840--1_bg.root", 176834],
        ["data176844--1_245_2764_3.root", "data176840--1_bg.root", 176844],
        ["data176850--1_245_3766_3.root", "data176840--1_bg.root", 176850],
        ["data176856--1_245_4768_3.root", "data176840--1_bg.root", 176856],
        ["data176862--1_245_5770_3.root", "data176840--1_bg.root", 176862],
        ["data176868--1_245_6772_3.root", "data176886--1_bg.root", 176868],
        ["data176874--1_245_7774_3.root", "data176886--1_bg.root", 176874],
        ["data176880--1_245_8776_3.root", "data176886--1_bg.root", 176880],
        ["data176890--1_320_2764_3.root", "data176886--1_bg.root", 176890],
        ["data176896--1_320_3766_3.root", "data176886--1_bg.root", 176896],
        ["data176902--1_320_4768_3.root", "data176886--1_bg.root", 176902],
        ["data176908--1_320_5770_3.root", "data176886--1_bg.root", 176908],
        ["data176914--1_320_6772_3.root", "data176932--1_bg.root", 176914],
        ["data176920--1_320_7774_3.root", "data176932--1_bg.root", 176920],
        ["data176926--1_320_8776_3.root", "data176932--1_bg.root", 176926],
    ]
)
scan_200119 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200119",
)
