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
        self.dir_sig_files = dir_sig_files
        self.dir_bg_files = dir_bg_files
        self.label = label
        self._measp_type = measp_type
        self._measp_src = measp_src

        self._sig_files = sig_files
        self._bg_files = bg_files
        self._positions = positions
        self._cyc_numbers = cyc_numbers
        self.events = None

    def __call__(self):
        if self.dir_sig_files[-1:] != "/":
            self.dir_sig_files = self.dir_sig_files + "/"
        if self.dir_bg_files is None:
            self.dir_bg_files = self.dir_sig_files
        if self.dir_bg_files[-1:] != "/":
            self.dir_bg_files = self.dir_bg_files + "/"

        ev_list = []
        for ind, sig_file in enumerate(self._sig_files):
            curr_ev = [self._measp_type, self._measp_src]
            curr_sig_file = self.dir_sig_files + sig_file
            curr_bg_file = self.dir_bg_files + self._bg_files[ind]
            curr_ev.append(list([curr_sig_file, curr_bg_file]))

            if self._cyc_numbers is not None:
                curr_ev.append(self._cyc_numbers[ind])
            ev_list.append(curr_ev)

        self.events = ev_list

        return self._positions, self.events

    def __str__(self):
        return "ScanFiles" + "_" + self.label

    def __repr__(self):
        return "ScanFiles" + "_" + self.label


pos_default = np.asarray(
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
        ["data135490--1_245_5770_3.root", "data135556.root", 135490],  # bad hist det1
        ["data135496--1_245_6772_3.root", "data135556.root", 135496],  # bad hist det1
        ["data135502--1_245_7774_3.root", "data135556.root", 135502],
        ["data135508--1_245_8776_3.root", "data135556.root", 135508],
        ["data135514--1_320_8776_3.root", "data135556.root", 135514],
        ["data135520--1_320_7774_3.root", "data135556.root", 135520],  # bad hist det1
        ["data135526--1_320_6772_3.root", "data135556.root", 135526],  # bad hist det1
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
    positions=pos_default,
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
        [170, 6772],
        [170, 8776],
        [170, 4768],
        [170, 2764],
        [170, 5770],
        [20, 6772],
        [20, 8776],
        [20, 4768],
        [20, 2764],
        [170, 5770],
        [320, 6772],
        [320, 8776],
        [320, 4768],
        [320, 2764],
        # [170, 5770],
    ]
)
evs = np.asarray(
    [
        ["data148313.root", "data148325.root", 148313],
        ["data148329.root", "data148341.root", 148329],
        ["data148345.root", "data148357.root", 148345],
        ["data148361.root", "data148373.root", 148361],
        # ["data148377.root", "data148389.root", 148377],
        ["data148393.root", "data148405.root", 148393],
        ["data148409.root", "data148421.root", 148409],
        ["data148425.root", "data148437.root", 148425],
        ["data148441.root", "data148453.root", 148441],
        ["data148457.root", "data148469.root", 148457],  # center
        ["data148473.root", "data148485.root", 148473],
        ["data148489.root", "data148501.root", 148489],
        ["data148505.root", "data148517.root", 148505],
        ["data148521.root", "data148533.root", 148521],
        # ["data148537.root", "data148549.root", 148537],
    ]
)
scan_200115 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200115",
)


evs = np.asarray(
    [
        ["data154644--1_20_2764_3.root", "data154640--1_bg.root", 154644],
        ["data154650--1_20_3766_3.root", "data154640--1_bg.root", 154650],
        ["data154656--1_20_4768_3.root", "data154640--1_bg.root", 154656],
        ["data154662--1_20_5770_3.root", "data154640--1_bg.root", 154662],
        ["data154668--1_20_6772_3.root", "data154686--1_bg.root", 154668],
        ["data154674--1_20_7774_3.root", "data154686--1_bg.root", 154674],
        ["data154680--1_20_8776_3.root", "data154686--1_bg.root", 154680],
        ["data154690--1_95_2764_3.root", "data154686--1_bg.root", 154690],
        ["data154696--1_95_3766_3.root", "data154686--1_bg.root", 154696],
        ["data154702--1_95_4768_3.root", "data154686--1_bg.root", 154702],
        ["data154708--1_95_5770_3.root", "data154686--1_bg.root", 154708],
        ["data154714--1_95_6772_3.root", "data154732--1_bg.root", 154714],
        ["data154720--1_95_7774_3.root", "data154732--1_bg.root", 154720],
        ["data154726--1_95_8776_3.root", "data154732--1_bg.root", 154726],
        ["data154736--1_170_2764_3.root", "data154732--1_bg.root", 154736],
        ["data154742--1_170_3766_3.root", "data154732--1_bg.root", 154742],
        ["data154748--1_170_4768_3.root", "data154732--1_bg.root", 154748],
        ["data154754--1_170_5770_3.root", "data154732--1_bg.root", 154754],
        ["data154760--1_170_6772_3.root", "data154778--1_bg.root", 154760],
        ["data154766--1_170_7774_3.root", "data154778--1_bg.root", 154766],
        ["data154772--1_170_8776_3.root", "data154778--1_bg.root", 154772],
        ["data154782--1_245_2764_3.root", "data154778--1_bg.root", 154782],
        ["data154788--1_245_3766_3.root", "data154778--1_bg.root", 154788],
        ["data154794--1_245_4768_3.root", "data154778--1_bg.root", 154794],
        ["data154800--1_245_5770_3.root", "data154778--1_bg.root", 154800],
        ["data154806--1_245_6772_3.root", "data154824--1_bg.root", 154806],
        ["data154812--1_245_7774_3.root", "data154824--1_bg.root", 154812],
        ["data154818--1_245_8776_3.root", "data154824--1_bg.root", 154818],
        ["data154828--1_320_2764_3.root", "data154824--1_bg.root", 154828],
        ["data154834--1_320_3766_3.root", "data154824--1_bg.root", 154834],
        ["data154840--1_320_4768_3.root", "data154824--1_bg.root", 154840],
        ["data154846--1_320_5770_3.root", "data154824--1_bg.root", 154846],
        ["data154852--1_320_6772_3.root", "data154870--1_bg.root", 154852],
        ["data154858--1_320_7774_3.root", "data154870--1_bg.root", 154858],
        ["data154864--1_320_8776_3.root", "data154870--1_bg.root", 154864],
    ]
)

scan_200116_3 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200116_sn",
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
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200117",
)

pos = np.asarray(
    [
        # [20, 2764],
        [20, 3766],
        [20, 4768],
        [20, 5770],
        [20, 6772],
        [20, 7774],
        [20, 8776],
        # [95, 2764],
        [95, 3766],
        [95, 4768],
        [95, 5770],
        [95, 6772],
        [95, 7774],
        [95, 8776],
        # [170, 2764],
        [170, 3766],
        [170, 4768],
        [170, 5770],
        [170, 6772],
        [170, 7774],
        [170, 8776],
        # [245, 2764],
        [245, 3766],
        [245, 4768],
        [245, 5770],
        [245, 6772],
        [245, 7774],
        [245, 8776],
        # [320, 2764],
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
        ["data171330--1_20_3766_3.root", "data171326--1_bg.root", 171330],
        ["data171336--1_20_4768_3.root", "data171326--1_bg.root", 171336],
        ["data171342--1_20_5770_3.root", "data171326--1_bg.root", 171342],
        ["data171348--1_20_6772_3.root", "data171366--1_bg.root", 171348],
        ["data171354--1_20_7774_3.root", "data171366--1_bg.root", 171354],
        ["data171360--1_20_8776_3.root", "data171366--1_bg.root", 171360],
        ["data171370--1_95_3766_3.root", "data171366--1_bg.root", 171370],
        ["data171376--1_95_4768_3.root", "data171366--1_bg.root", 171376],
        ["data171382--1_95_5770_3.root", "data171366--1_bg.root", 171382],
        ["data171388--1_95_6772_3.root", "data171406--1_bg.root", 171388],
        ["data171394--1_95_7774_3.root", "data171406--1_bg.root", 171394],
        ["data171400--1_95_8776_3.root", "data171406--1_bg.root", 171400],
        ["data171410--1_170_3766_3.root", "data171406--1_bg.root", 171410],
        ["data171416--1_170_4768_3.root", "data171406--1_bg.root", 171416],
        ["data171422--1_170_5770_3.root", "data171406--1_bg.root", 171422],
        ["data171428--1_170_6772_3.root", "data171446--1_bg.root", 171428],
        ["data171434--1_170_7774_3.root", "data171446--1_bg.root", 171434],
        ["data171440--1_170_8776_3.root", "data171446--1_bg.root", 171440],
        ["data171450--1_245_3766_3.root", "data171446--1_bg.root", 171450],
        ["data171456--1_245_4768_3.root", "data171446--1_bg.root", 171456],
        ["data171462--1_245_5770_3.root", "data171446--1_bg.root", 171462],
        ["data171468--1_245_6772_3.root", "data171486--1_bg.root", 171468],
        ["data171474--1_245_7774_3.root", "data171486--1_bg.root", 171474],
        ["data171480--1_245_8776_3.root", "data171486--1_bg.root", 171480],
        ["data171490--1_320_3766_3.root", "data171486--1_bg.root", 171490],
        ["data171496--1_320_4768_3.root", "data171486--1_bg.root", 171496],
        ["data171502--1_320_5770_3.root", "data171486--1_bg.root", 171502],
        ["data171508--1_320_6772_3.root", "data171526--1_bg.root", 171508],
        ["data171514--1_320_7774_3.root", "data171526--1_bg.root", 171514],
        ["data171520--1_320_8776_3.root", "data171526--1_bg.root", 171520],
    ]
)
scan_200118 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos,
    cyc_numbers=evs.T[2],
    label="sn_200118",
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
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200119",
)


evs = np.asarray(
    [
        ["data182345--1_20_2764_3.root", "data182341--1_bg.root", 182345],
        ["data182351--1_20_3766_3.root", "data182341--1_bg.root", 182351],
        ["data182357--1_20_4768_3.root", "data182341--1_bg.root", 182357],
        ["data182363--1_20_5770_3.root", "data182341--1_bg.root", 182363],
        ["data182369--1_20_6772_3.root", "data182387--1_bg.root", 182369],
        ["data182375--1_20_7774_3.root", "data182387--1_bg.root", 182375],
        ["data182381--1_20_8776_3.root", "data182387--1_bg.root", 182381],
        ["data182391--1_95_2764_3.root", "data182387--1_bg.root", 182391],
        ["data182397--1_95_3766_3.root", "data182387--1_bg.root", 182397],
        ["data182403--1_95_4768_3.root", "data182387--1_bg.root", 182403],
        ["data182409--1_95_5770_3.root", "data182387--1_bg.root", 182409],
        ["data182415--1_95_6772_3.root", "data182433--1_bg.root", 182415],
        ["data182421--1_95_7774_3.root", "data182433--1_bg.root", 182421],
        ["data182427--1_95_8776_3.root", "data182433--1_bg.root", 182427],
        ["data182437--1_170_2764_3.root", "data182433--1_bg.root", 182437],
        ["data182443--1_170_3766_3.root", "data182433--1_bg.root", 182443],
        ["data182449--1_170_4768_3.root", "data182433--1_bg.root", 182449],
        ["data182455--1_170_5770_3.root", "data182433--1_bg.root", 182455],
        ["data182461--1_170_6772_3.root", "data182479--1_bg.root", 182461],
        ["data182467--1_170_7774_3.root", "data182479--1_bg.root", 182467],
        ["data182473--1_170_8776_3.root", "data182479--1_bg.root", 182473],
        ["data182483--1_245_2764_3.root", "data182479--1_bg.root", 182483],
        ["data182489--1_245_3766_3.root", "data182479--1_bg.root", 182489],
        ["data182495--1_245_4768_3.root", "data182479--1_bg.root", 182495],
        ["data182501--1_245_5770_3.root", "data182479--1_bg.root", 182501],
        ["data182507--1_245_6772_3.root", "data182525--1_bg.root", 182507],
        ["data182513--1_245_7774_3.root", "data182525--1_bg.root", 182513],
        ["data182519--1_245_8776_3.root", "data182525--1_bg.root", 182519],
        ["data182529--1_320_2764_3.root", "data182525--1_bg.root", 182529],
        ["data182535--1_320_3766_3.root", "data182525--1_bg.root", 182535],
        ["data182541--1_320_4768_3.root", "data182525--1_bg.root", 182541],
        ["data182547--1_320_5770_3.root", "data182525--1_bg.root", 182547],
        ["data182553--1_320_6772_3.root", "data182571--1_bg.root", 182553],
        ["data182559--1_320_7774_3.root", "data182571--1_bg.root", 182559],
        ["data182565--1_320_8776_3.root", "data182571--1_bg.root", 182565],
    ]
)
scan_200120 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200120",
)

evs = np.asarray(
    [
        ["data189320--1_20_2764_3.root", "data189316--1_bg.root", 189320],
        ["data189326--1_20_3766_3.root", "data189316--1_bg.root", 189326],
        ["data189332--1_20_4768_3.root", "data189316--1_bg.root", 189332],
        ["data189338--1_20_5770_3.root", "data189316--1_bg.root", 189338],
        ["data189344--1_20_6772_3.root", "data189362--1_bg.root", 189344],
        ["data189350--1_20_7774_3.root", "data189362--1_bg.root", 189350],
        ["data189356--1_20_8776_3.root", "data189362--1_bg.root", 189356],
        ["data189366--1_95_2764_3.root", "data189362--1_bg.root", 189366],
        ["data189372--1_95_3766_3.root", "data189362--1_bg.root", 189372],
        ["data189378--1_95_4768_3.root", "data189362--1_bg.root", 189378],
        ["data189384--1_95_5770_3.root", "data189362--1_bg.root", 189384],
        ["data189390--1_95_6772_3.root", "data189408--1_bg.root", 189390],
        ["data189396--1_95_7774_3.root", "data189408--1_bg.root", 189396],
        ["data189402--1_95_8776_3.root", "data189408--1_bg.root", 189402],
        ["data189412--1_170_2764_3.root", "data189408--1_bg.root", 189412],
        ["data189418--1_170_3766_3.root", "data189408--1_bg.root", 189418],
        ["data189424--1_170_4768_3.root", "data189408--1_bg.root", 189424],
        ["data189430--1_170_5770_3.root", "data189408--1_bg.root", 189430],
        ["data189436--1_170_6772_3.root", "data189454--1_bg.root", 189436],
        ["data189442--1_170_7774_3.root", "data189454--1_bg.root", 189442],
        ["data189448--1_170_8776_3.root", "data189454--1_bg.root", 189448],
        ["data189458--1_245_2764_3.root", "data189454--1_bg.root", 189458],
        ["data189464--1_245_3766_3.root", "data189454--1_bg.root", 189464],
        ["data189470--1_245_4768_3.root", "data189454--1_bg.root", 189470],
        ["data189476--1_245_5770_3.root", "data189454--1_bg.root", 189476],
        ["data189482--1_245_6772_3.root", "data189500--1_bg.root", 189482],
        ["data189488--1_245_7774_3.root", "data189500--1_bg.root", 189488],
        ["data189494--1_245_8776_3.root", "data189500--1_bg.root", 189494],
        ["data189504--1_320_2764_3.root", "data189500--1_bg.root", 189504],
        ["data189510--1_320_3766_3.root", "data189500--1_bg.root", 189510],
        ["data189516--1_320_4768_3.root", "data189500--1_bg.root", 189516],
        ["data189522--1_320_5770_3.root", "data189500--1_bg.root", 189522],
        ["data189528--1_320_6772_3.root", "data189546--1_bg.root", 189528],
        ["data189534--1_320_7774_3.root", "data189546--1_bg.root", 189534],
        ["data189540--1_320_8776_3.root", "data189546--1_bg.root", 189540],
    ]
)
scan_200121 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200121",
)


evs = np.asarray(
    [
        ["data196540--1_20_2764_3.root", "data196004--1_bg.root", 196540],
        ["data196546--1_20_3766_3.root", "data196004--1_bg.root", 196546],
        ["data196552--1_20_4768_3.root", "data196004--1_bg.root", 196552],
        ["data196558--1_20_5770_3.root", "data196004--1_bg.root", 196558],
        ["data196564--1_20_6772_3.root", "data196050--1_bg.root", 196564],
        ["data196570--1_20_7774_3.root", "data196050--1_bg.root", 196570],
        ["data196576--1_20_8776_3.root", "data196050--1_bg.root", 196576],
        ["data196586--1_95_2764_3.root", "data196050--1_bg.root", 196586],
        ["data196592--1_95_3766_3.root", "data196050--1_bg.root", 196592],
        ["data196598--1_95_4768_3.root", "data196050--1_bg.root", 196598],
        ["data196604--1_95_5770_3.root", "data196050--1_bg.root", 196604],
        ["data196610--1_95_6772_3.root", "data196096--1_bg.root", 196610],
        ["data196616--1_95_7774_3.root", "data196096--1_bg.root", 196616],
        ["data196622--1_95_8776_3.root", "data196096--1_bg.root", 196622],
        ["data196636--1_170_2764_3.root", "data196096--1_bg.root", 196636],
        ["data196642--1_170_3766_3.root", "data196096--1_bg.root", 196642],
        ["data196648--1_170_4768_3.root", "data196096--1_bg.root", 196648],
        ["data196654--1_170_5770_3.root", "data196096--1_bg.root", 196654],
        ["data196660--1_170_6772_3.root", "data196142--1_bg.root", 196660],
        ["data196666--1_170_7774_3.root", "data196142--1_bg.root", 196666],
        ["data196672--1_170_8776_3.root", "data196142--1_bg.root", 196672],
        ["data196682--1_245_2764_3.root", "data196142--1_bg.root", 196682],
        ["data196688--1_245_3766_3.root", "data196142--1_bg.root", 196688],
        ["data196694--1_245_4768_3.root", "data196142--1_bg.root", 196694],
        ["data196700--1_245_5770_3.root", "data196142--1_bg.root", 196700],
        ["data196706--1_245_6772_3.root", "data196188--1_bg.root", 196706],
        ["data196712--1_245_7774_3.root", "data196188--1_bg.root", 196712],
        ["data196718--1_245_8776_3.root", "data196188--1_bg.root", 196718],
        ["data196728--1_320_2764_3.root", "data196188--1_bg.root", 196728],
        ["data196734--1_320_3766_3.root", "data196188--1_bg.root", 196734],
        ["data196740--1_320_4768_3.root", "data196188--1_bg.root", 196740],
        ["data196746--1_320_5770_3.root", "data196188--1_bg.root", 196746],
        ["data196752--1_320_6772_3.root", "data196234--1_bg.root", 196752],
        ["data196758--1_320_7774_3.root", "data196234--1_bg.root", 196758],
        ["data196764--1_320_8776_3.root", "data196234--1_bg.root", 196764],
    ]
)
scan_200122 = ScanFiles(
    dir_sig_files=DEF_SIG_DIR,
    dir_bg_files=DEF_BG_DIR,
    sig_files=evs.T[0],
    bg_files=evs.T[1],
    positions=pos_default,
    cyc_numbers=evs.T[2],
    label="sn_200122",
)
