"""Module for handling Perkeo root files and creating histograms."""

import glob
import os
import pickle

import numpy as np
import pandas as pd

output_path = os.getcwd()


def ret_hist(
    data: np.array(float),
    bin_count: int = 1024,
    low_lim: int = 0,
    up_lim: int = 52000,
):
    """Create a histogram as pd.DataFrame from an input array."""

    raw_bins = np.linspace(low_lim, up_lim, bin_count + 1)
    use_bins = [np.array([-np.inf]), raw_bins, np.array([np.inf])]
    use_bins = np.concatenate(use_bins)

    hist, binedge = np.histogram(data, bins=use_bins)

    bincent = []
    for j in range(binedge.size - 1):
        bincent.append(0.5 * (binedge[j] + binedge[j + 1]))

    hist = hist[1:-1]
    bincent = bincent[1:-1]

    return pd.DataFrame({"x": bincent, "y": hist, "err": np.sqrt(np.abs(hist))})


def ret_hist2d(
    data: np.array,
    bin_count=None,
    low_lim=None,
    up_lim=None,
):
    """Create a 2D histogram as dict from a 2D input array."""

    if up_lim is None:
        up_lim = [52000.0, 52000.0]
    if low_lim is None:
        low_lim = [0.0, 0.0]
    if bin_count is None:
        bin_count = [1024, 1024]
    assert data.shape[0] == 2, f"Wrong data dimension. {data.shape[0]} != 2"
    raw_binsx = np.linspace(low_lim[0], up_lim[0], bin_count[0] + 1)
    raw_binsy = np.linspace(low_lim[1], up_lim[1], bin_count[1] + 1)

    use_binsx = [np.array([-np.inf]), raw_binsx, np.array([np.inf])]
    use_binsx = np.concatenate(use_binsx)
    use_binsy = [np.array([-np.inf]), raw_binsy, np.array([np.inf])]
    use_binsy = np.concatenate(use_binsy)

    hist, binedgex, binedgey = np.histogram2d(
        data[0], data[1], bins=(use_binsx, use_binsy)
    )
    bincentx = []
    for j in range(binedgex.size - 1):
        bincentx.append(0.5 * (binedgex[j] + binedgex[j + 1]))

    bincenty = []
    for j in range(binedgey.size - 1):
        bincenty.append(0.5 * (binedgey[j] + binedgey[j + 1]))

    # For cartesian convention
    hist = hist.T
    # remove over- and underflow bins
    res = {
        "x": bincentx[1:-1],
        "y": bincenty[1:-1],
        "x_edges": binedgex[1:-1],
        "y_edges": binedgey[1:-1],
        "z": hist[1:-1, 1:-1],
        "err": np.sqrt(np.abs(hist))[1:-1, 1:-1],
    }

    return res


def filt_zeros(hist_df: pd.DataFrame, bdrop_nan: bool = True) -> pd.DataFrame:
    """Taking a pandas data frame and removing all entries where "err" = 0."""

    filt = hist_df["err"] != 0.0
    hist_df = hist_df[filt]
    if bdrop_nan:
        hist_df = hist_df.dropna()

    assert not hist_df["x"].isnull().values.any()
    assert not hist_df["y"].isnull().values.any()
    assert not hist_df["err"].isnull().values.any()

    return hist_df


def concat_hists(hist_array: np.array):
    """Concatenate multiple histograms in an array by adding them up with error prop."""

    hist_final = hist_array[0]
    for hist in hist_array[1:]:
        hist_final.addhist(hist)

    return hist_final


class FilePerkeo:
    """Obj for writing/reading data from/into a general binary file."""

    def __init__(self, filename: str):
        self.filename = filename

    def imp(self):
        """Open the file and return content."""
        with open(self.filename, "rb") as file:
            imp_obj = pickle.load(file)

        return imp_obj

    # FIXME: Append doesn't append? Probs because import only expects one object
    def dump(self, obj, out_dir: str = None, bapp: bool = False, btext: bool = False):
        """Dump an python object into the file.

        Parameters
        ----------
        obj
        out_dir
        bapp : bool
            Append to file instead of writing over it. Doesn't work.
            (default False)
        btext : bool
            Bool whether to dump obj as binary or text into file.
            (default False)
        """

        if out_dir is None:
            out_dir = output_path

        opt = "a" if bapp else "w"
        if not btext:
            with open(f"{out_dir}/{self.filename}", opt + "b") as file:
                pickle.dump(obj, file)
        else:
            with open(f"{out_dir}/{self.filename}", opt) as file:
                file.write(obj)

        return 0


class DirPerkeo:
    """Obj for extracting file name lists out of directory."""

    def __init__(self, dirname: str, filetype: str = "root"):
        self.dirname = dirname
        self.filetype = filetype

    def get_all(self, bsorted=True) -> list:
        """Returns list of all files of given type in directory"""

        if bsorted:
            liste = sorted(glob.glob(self.dirname + "*." + self.filetype))
        else:
            liste = glob.glob(self.dirname + "*." + self.filetype)

        return liste

    def get_subset(self, liste: list) -> list:
        """Returns list of required files (full name) out of list in dir"""

        retlist = []
        for i in liste:
            retlist.append(glob.glob(self.dirname + i)[0])

        return retlist


class FiltPerkeo:
    """Obj for creating/setting filters for RootPerkeo Trees.

    Attributes
    ----------
    active : bool
        Status bool whether filter is active
    type : str
        Filter type for safety handling: 'bool', 'num' or special
        type 'cyc'
    low_lim, up_lim: float
        Filter range for 'num' type filter
    rightval
        Value for 'bool' type filter to check quality for
    index
        index to be used, e.g. CoinTime[index]
    """

    def __init__(self, low_lim=None, up_lim=None, rightval=None, index=None, **kwargs):
        self.active = kwargs["active"]
        self.ftype = kwargs["ftype"]
        self.fkey = kwargs["fkey"]
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.rightval = rightval
        self.index = index
