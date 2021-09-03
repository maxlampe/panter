"""This module should manage filelists and setup further evaluations. """

import datetime
import sys

import numpy as np

import panter.config.filesElecTest as fET
from panter.data.dataPerkeo import DirPerkeo


def is_integer(number_string) -> bool:
    """Test whether number or string is actually integer."""
    try:
        float(number_string)
    except ValueError:
        return False
    else:
        return float(number_string).is_integer()


def sort_files(curr_dir: str):
    """Setting up drift evaluation settings.

    Function for setting up drift evaluation settings like output
    filename, getting file lists etc.

    Parameters
    ----------
    curr_dir : str

    Returns
    -------
    measlist : list of str

    """

    curr_dir = DirPerkeo(curr_dir)

    list_all_rootfiles = list(
        map(lambda x: x[len(curr_dir.dirname) : -5], curr_dir.get_all())
    )
    list_all_backfiles = list(filter(lambda x: x[-2:] == "bg", list_all_rootfiles))
    list_all_beamfiles = list(filter(lambda x: x[-4:] == "beam", list_all_rootfiles))
    list_all_sorcfiles = list(filter(lambda x: is_integer(x[-1]), list_all_rootfiles))

    measlist = []

    for file in list_all_beamfiles:
        currnomad_it = 0
        currcycle_it = 0
        if is_integer(file[-10:-5]):
            currnomad_it = int(file[-10:-5])
        if is_integer(file[4:10]):
            currcycle_it = int(file[4:10])
        if currnomad_it > 1 and currcycle_it > 0:
            measlist.append(
                [
                    0,
                    5,
                    [curr_dir.dirname + file + ".root"],
                    currcycle_it,
                    currnomad_it,
                ]
            )
        else:
            print("WARNING: Invalid cycle or NOMAD number.")

    for file in list_all_sorcfiles:
        currnomad_it = 0
        currcycle_it = 0
        if is_integer(file[-7:-2]):
            currnomad_it = int(file[-7:-2])
        if is_integer(file[4:10]):
            currcycle_it = int(file[4:10])
        if currnomad_it > 1 and currcycle_it > 0:
            source_type = int(file[-1])
            for file_bg in list_all_backfiles:
                if is_integer(file_bg[-8:-3]) and int(file_bg[-8:-3]) == currnomad_it:
                    name_of_pair = [
                        curr_dir.dirname + file + ".root",
                        curr_dir.dirname + file_bg + ".root",
                    ]
                    measlist.append(
                        [
                            1,
                            source_type,
                            name_of_pair,
                            currcycle_it,
                            currnomad_it,
                        ]
                    )

    # measlist = np.array(measlist)

    return measlist


def setupeval_electest(
    mtype: str = "Test4",
    fit_mode: int = 0,
    fanout_mode: int = 0,
    bfitall: bool = False,
    bfixsig: bool = False,
    bfixnorm: bool = False,
):
    """Setting up electronics test evaluation settings.

    Function for setting up electronics test evaluation settings like
    output filename, getting file lists etc.

    Parameters
    ----------
    mtype : {'Test4', 'Test5'}
        String argument to indicate which data will be analysed after
        the setup. (default 'Test4')
    fit_mode : {0, 1, 2}
        Int parameter for function/model to be fitted.
        0 = gaus_gen
        1 = gaus_DoGG
        2 = gaus_simp
        in evalFitSettings.py. (default 0)
    fanout_mode : {0, 1, 2, 3, 4}
        Int parameter to chose FanOut module (has effect on imported
        raw data). If equal 4, all modules are evaluated at the same
        time.
    bfitall : bool
        If True, all 16 PMT specs are fitted. Only 'active' ones
        otherwise. (default False)
    bfixsig, bfixnorm : bool
        Bool to fix either Sigma or Norm parameter in Gaussian fits.
        (default False)

    Returns
    -------
    filelist: list of str
    results: list of None
        Empty list of len(filelist) to be filled with results later.
    fit_mode : int
        Is equal to input parameter fit_mode.
    expfile : str
        File name for evaluation results to be written into.
    blist: list of bools
    """

    blist = [bfitall, bfixsig, bfixnorm]

    ev_mode = int(fanout_mode)
    if (np.array([0, 1, 2, 3, 4]) == ev_mode).sum() <= 0:
        print("ERROR: Wrong FanOut selection input!")
        sys.exit(1)

    fit_mode = int(fit_mode)
    if (np.array([0, 1, 2]) == fit_mode).sum() <= 0:
        print("ERROR: Wrong fit_module input!")
        sys.exit(1)

    if blist[1]:
        str_fs = "bfixsig_"
    else:
        str_fs = ""
    if blist[2]:
        str_fn = "FixNor_"
    else:
        str_fn = ""

    if fit_mode == 0:
        str_type = "SG_"
    if fit_mode == 1:
        str_type = "DoGG_"
    if fit_mode == 2:
        str_type = "RawGaussian_"

    if mtype == "Test4":
        curr_dir = DirPerkeo(fET.TEST4_DIR)

        filelist1 = curr_dir.get_subset(fET.Test4_0003_200)
        filelist2 = curr_dir.get_subset(fET.Test4_0003_300)
        filelist3 = curr_dir.get_subset(fET.Test4_0003_500)

        filelist4 = curr_dir.get_subset(fET.Test4_0407_200)
        filelist5 = curr_dir.get_subset(fET.Test4_0407_300)
        filelist6 = curr_dir.get_subset(fET.Test4_0407_500)

        filelist7 = curr_dir.get_subset(fET.Test4_0811_200)
        filelist8 = curr_dir.get_subset(fET.Test4_0811_300)
        filelist9 = curr_dir.get_subset(fET.Test4_0811_500)

        filelist10 = curr_dir.get_subset(fET.Test4_1215_200)
        filelist11 = curr_dir.get_subset(fET.Test4_1215_300)
        filelist12 = curr_dir.get_subset(fET.Test4_1215_500)

        if ev_mode == 0:
            filelist = [filelist1, filelist2, filelist3]
            results = [
                [None] * len(filelist1),
                [None] * len(filelist2),
                [None] * len(filelist3),
            ]
            str_ev = "0003_"
        if ev_mode == 1:
            filelist = [filelist4, filelist5, filelist6]
            results = [
                [None] * len(filelist4),
                [None] * len(filelist5),
                [None] * len(filelist6),
            ]
            str_ev = "0407_"
        if ev_mode == 2:
            filelist = [filelist7, filelist8, filelist9]
            results = [
                [None] * len(filelist7),
                [None] * len(filelist8),
                [None] * len(filelist9),
            ]
            str_ev = "0811_"
        if ev_mode == 3:
            filelist = [filelist10, filelist11, filelist12]
            results = [
                [None] * len(filelist10),
                [None] * len(filelist11),
                [None] * len(filelist12),
            ]
            str_ev = "1215_"
        if ev_mode == 4:
            filelist = [
                filelist1,
                filelist2,
                filelist3,
                filelist4,
                filelist5,
                filelist6,
                filelist7,
                filelist8,
                filelist9,
                filelist10,
                filelist11,
                filelist12,
            ]
            results = [
                [None] * len(filelist1),
                [None] * len(filelist2),
                [None] * len(filelist3),
                [None] * len(filelist4),
                [None] * len(filelist5),
                [None] * len(filelist6),
                [None] * len(filelist7),
                [None] * len(filelist8),
                [None] * len(filelist9),
                [None] * len(filelist10),
                [None] * len(filelist11),
                [None] * len(filelist12),
            ]
            str_ev = "ALL_"
    else:
        if mtype == "Test5":
            curr_dir = DirPerkeo(fET.TEST5_DIR)

            filelist1 = curr_dir.get_subset(fET.Test5_0003_200500)
            filelist2 = curr_dir.get_subset(fET.Test5_0003_050300)

            filelist3 = curr_dir.get_subset(fET.Test5_0407_200500)
            filelist4 = curr_dir.get_subset(fET.Test5_0407_050300)

            filelist5 = curr_dir.get_subset(fET.Test5_0811_200500)
            filelist6 = curr_dir.get_subset(fET.Test5_0811_050300)

            filelist7 = curr_dir.get_subset(fET.Test5_1215_200500)
            filelist8 = curr_dir.get_subset(fET.Test5_1215_050300)

            if ev_mode == 0:
                filelist = [filelist1, filelist2]
                results = [[None] * len(filelist1), [None] * len(filelist2)]
                str_ev = "0003_"
            if ev_mode == 1:
                filelist = [filelist3, filelist4]
                results = [[None] * len(filelist3), [None] * len(filelist4)]
                str_ev = "0407_"
            if ev_mode == 2:
                filelist = [filelist5, filelist6]
                results = [[None] * len(filelist5), [None] * len(filelist6)]
                str_ev = "0811_"
            if ev_mode == 3:
                filelist = [filelist7, filelist8]
                results = [[None] * len(filelist7), [None] * len(filelist8)]
                str_ev = "1215_"
            if ev_mode == 4:
                filelist = [
                    filelist1,
                    filelist2,
                    filelist3,
                    filelist4,
                    filelist5,
                    filelist6,
                    filelist7,
                    filelist8,
                ]
                results = [
                    [None] * len(filelist1),
                    [None] * len(filelist2),
                    [None] * len(filelist3),
                    [None] * len(filelist4),
                    [None] * len(filelist5),
                    [None] * len(filelist6),
                    [None] * len(filelist7),
                    [None] * len(filelist8),
                ]
                str_ev = "ALL_"
        else:
            print("ERROR: Unavailable Test setup! \t", mtype)
            sys.exit(1)

    cdir = datetime.date.today()
    expfile = mtype + "_" + str_ev + str_type + str_fs + str_fn + str(cdir) + ".p"

    return filelist, results, fit_mode, expfile, blist
