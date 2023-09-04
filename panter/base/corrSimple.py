"""Module for correcting PERKEO data for a general filter."""

import numpy as np

from panter.base.corrBase import CorrBase
from panter.data.dataHistPerkeo import HistPerkeo
from panter.data.dataRootPerkeo import RootPerkeo


class CorrSimple(CorrBase):
    """Class for doing simple correction on PERKEO data.

    Takes a data loader and corrects all with background subtraction and dead time.
    Generally used for non-Energy spectra, e.g., time-of-flight data. Inherits
    from CorrBase.

    Parameters
    ----------
    dataloader: np.array()
        Array of measurements to be corrected created with data loader.
    bonlynew: True
        Only create corrected spectra instead of uncorrected spectra as well.
    hist_par: {"bin_count": 1000, "low_lim": 0, "up_lim": 10000}
        Parameters to be used for histogram (HistPerkeo).
    branch_key: str
        Key of branch in data tree to be used, e.g., CoinTime.

    Attributes
    ----------
    corrections : {
        "DeadTime": True,
    }
        Dictionary with bools to turn on/off different data corrections.
    See base class for other attributes.

    Examples
    --------
    Can be used to just calculate background subtracted data. If corrections were to be
    set to True, data would be individually corrected and then background subtracted.
    Corrections can be all turned off or on with the shown class method.

    >>> meas = DLPerkeo().ret_meas()
    >>> corr_class = CorrSimple(meas)
    >>> corr_class.set_all_corr(bactive=True)
    >>> corr_class.corr(bstore=True, bwrite=False)
    """

    def __init__(
        self,
        dataloader: np.array,
        branch_key: str,
        hist_par: dict = None,
        bonlynew: bool = True,
    ):
        super().__init__(dataloader=dataloader, bonlynew=bonlynew)
        self._branch_key = branch_key
        self._hist_par = hist_par
        if self._hist_par is None:
            self._hist_par = {"bin_count": 1000, "low_lim": 0, "up_lim": 10000}
        self.corrections = {
            "DeadTime": True,
        }

    def _calc_corr(self, data: RootPerkeo, bdiff: bool = True):
        """Calculate corrected amplitude for each event and file.

        Parameters
        ----------
        data: RootPerkeo
            Data class to be corrected.
        bdiff: True
            Assumes 2D events. Calculates difference between the two values. Mainly for
            CoinTime to get DeltaCoinTime.
        """

        binvalid = False
        data_by_branch = data.ret_array_by_key(key=self._branch_key)
        if len(data_by_branch.shape) == 1:
            hist_of_branch = [HistPerkeo(data_by_branch, **self._hist_par)]
        else:
            data_by_branch = data_by_branch.transpose()
            if bdiff:
                coin0 = np.array(data_by_branch[0], dtype=float)
                coin1 = np.array(data_by_branch[1], dtype=float)
                diff = coin0 - coin1
                hist_of_branch = [HistPerkeo(diff, **self._hist_par)]
            else:
                hist_of_branch = []
                for dim in data_by_branch:
                    hist_of_branch.append(HistPerkeo(dim, **self._hist_par))

        if self._bonlynew:
            hist_old = None
        else:
            hist_old = hist_of_branch
        hist_new = hist_of_branch

        if self.corrections["DeadTime"]:
            datacop = RootPerkeo(data.filename)
            datacop.set_filtdef()
            datacop.auto()

            for hist in range(len(hist_new)):
                if not self._bonlynew:
                    hist_old[hist].scal(datacop.dt_fac)
                hist_new[hist].scal(datacop.dt_fac)

        return [[hist_old, hist_new], data.cy_valid_no, binvalid]

    def _corr_nobg(self, ev_file: list):
        """Correct measurement without background subtraction

        Parameters
        ----------
        ev_file: list
            File name of data file as a list with one entry.
        """

        res = []
        data_sg = RootPerkeo(ev_file[0])

        self._filt_data(data_sg)
        r, s, i = self._calc_corr(data_sg)
        if i:
            return [None, None]
        res.append(r)

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def _corr_beam(self, ev_file: list):
        """Correct beam-like data (i.e. background in same file).

        Parameters
        ----------
        ev_file: list
            File name of data file as a list with one entry.
        """

        res = []
        data_sg = RootPerkeo(ev_file[0])
        data_bg = RootPerkeo(ev_file[0])
        data_dict = {"sg": data_sg, "bg": data_bg}

        for key, data in data_dict.items():
            self._filt_data(data, bbeam=True, key=key)
            r, s, i = self._calc_corr(data)
            if i:
                return [None, None]
            res.append(r)

        fac = [
            (self._beam_mtime["sg"][1] - self._beam_mtime["sg"][0])
            / (self._beam_mtime["bg"][1] - self._beam_mtime["bg"][0])
        ] * 2

        for hist_no in range(len(res[0][1])):
            if not self._bonlynew:
                res[0][0][hist_no].addhist(res[1][0][hist_no], -fac[0])
            res[0][1][hist_no].addhist(res[1][1][hist_no], -fac[1])

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def _corr_src(self, ev_files: list):
        """Correct source-like data (i.e. background in different file).

        Parameters
        ----------
        ev_files: list
            Signal and background file names in a list.
        """

        res = []
        scal = []

        for file_name in ev_files:
            data = RootPerkeo(file_name)
            self._filt_data(data)

            r, s, i = self._calc_corr(data)
            if i:
                return [None, None]
            res.append(r)
            scal.append(s)

        for hist_no in range(len(res[0][1])):
            if not self._bonlynew:
                res[0][0][hist_no].addhist(res[1][0][hist_no], -scal[0] / scal[1])
            res[0][1][hist_no].addhist(res[1][1][hist_no], -scal[0] / scal[1])

        if self._bonlynew:
            res_old = None
        else:
            res_old = [*res[0][0]]
        res_new = [*res[0][1]]

        return [res_old, res_new]

    def corr(
        self,
        bstore: bool = False,
        bwrite: bool = True,
        bconcat: bool = False,
        label: str = None,
    ):
        """Correcting data according to chosen settings.

        Parameters
        ----------
        bstore: False
            Bool whether to append created histograms in self.histograms
        bwrite: True
            Bool whether to write created histograms to a ROOT file.
        bconcat: False
            Bool whether to concatenate spectra.
        label: "CorrSimp"
            Default file name prefix.
        """

        if not bwrite and not bstore:
            print("WARNING: Doing nothing with data")

        if label is None:
            label = "CorrSimp"
        corr = ""
        for corr_name, is_active in self.corrections.items():
            if is_active:
                corr += corr_name
        cyc_no = 0

        # Catch single MeasPerkeo inputs
        try:
            self._dataloader.shape
        except AttributeError:
            self._dataloader = [self._dataloader]

        for meas in self._dataloader:
            tp = meas.tp
            src = meas.src
            files = meas.file_list
            if meas.cyc_no is not None:
                cyc_no = meas.cyc_no
            else:
                cyc_no += 1

            src_name = "PerkeoHist"
            if src == 5:
                src_name = "Beam"
            if (np.array([0, 1, 2, 3, 4]) == src).sum() < 0:
                src_name = f"Src{src}"

            if tp == 0:
                [hist_o, hist_n] = self._corr_beam(files)
            elif tp == 1:
                [hist_o, hist_n] = self._corr_src(files)
            elif tp == 2:
                [hist_o, hist_n] = self._corr_nobg(files)

            if bwrite:
                filename = f"{label}_{src_name}_{cyc_no}_{corr}.root"
                hist_n[0].write2root(histname=f"DetSumTot", filename=filename)
                if not self._bonlynew:
                    hist_o[0].write2root(
                        histname=f"DetSumTot", filename=filename, bupdate=True
                    )

            if bconcat:
                if self.hist_concat is None:
                    if hist_n is not None:
                        self.hist_concat = hist_n[0]
                if self.hist_concat is not None:
                    if hist_n is not None:
                        self.hist_concat.addhist(hist_n[0])

            if bstore:
                if hist_o is None:
                    hist_o = [None] * len(hist_n)
                self.histograms.append(np.asarray([hist_o, hist_n]))

        self.histograms = np.asarray(self.histograms)

        return 0
