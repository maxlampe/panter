""""""

import os


class MeasPerkeo:
    """Measurement class for the data loader. If called, returns measurement as list.

    Parameters
    ----------
    tp : int
        States whether beam-like (0), source-like data(1) or without background(2), for background subtraction.
    src : int
        Measurment type. Convention: 0-4 calibration sources,
        5 beam, 6-7 background w/wo B field and 8 would be electronic tests.
    file_list : list
        List of file(s) for single measurement. One file for tp = 0 and 2, two for tp = 1.
    cyc_no, nomad_no : int
        Optional.

    Attributes
    ----------
    tp, src : int
    file_list : list of str
    date_list : list of float
        List of file modification time from file_list.
    cyc_no, nomad_no : int
    """

    def __init__(
        self,
        tp: int,
        src: int,
        file_list: list,
        cyc_no: int = None,
        nomad_no: int = None,
    ):
        self.tp = tp
        self.src = src
        self.file_list = file_list
        self.date_list = self._getdates()
        self.cyc_no = cyc_no
        self.nomad_no = nomad_no

    def __str__(self):
        return "MeasP" + str(self.cyc_no) + "_src_" + str(self.src)

    def __repr__(self):
        return "MeasP" + str(self.cyc_no) + "_src_" + str(self.src)

    def __call__(self, *args, **kwargs):
        event = []
        for (key, entry) in self.__dict__.items():
            event.append(entry)
        return event

    def _getdates(self):
        """Get list of modification times for entries in file_list."""

        dates = []
        for file_name in self.file_list:
            dates.append(os.path.getmtime(file_name))

        return dates
