from panter.data.dataRootPerkeo import RootPerkeo
import numpy as np

# I strongly recommend checking out the docstring and the examples for RootPerkeo class
# call via running
# >>> help(RootPerkeo)

# -------- Quickstart

# Initialize the RootPerkeo object with a ROOT file from PERKEO III data
data = RootPerkeo("example.root")

# auto() method applies default settings and filters to process the data
data.auto()

# Generate histograms for all sixteen PMTs used in PERKEO III (generated by index)
data.gen_hist(list(range(16)))

# Plot histogram for the first PMT to visually inspect the data
data.hists[0].plot_hist()

# -------- Alternatively

data = RootPerkeo("example.root")

# Apply custom filters to the data. Filters are used to refine the data
# based on certain criteria, such as time differences and detector responses
# Here, we set specific filters for data fields like DeltaTriggerTime and Detector
data.set_filtdef()
data.set_filt(
    tree="data",
    fkey="DeltaTriggerTime",
    active=True,
    ftype="num",
    low_lim=380000,
    up_lim=600000,
)
data.set_filt(tree="data", fkey="Detector", active=True, ftype="bool", rightval=0)

# Event-based corrections dict specifies which corrections to apply to the PMT data
# Core correction class is CorrPerkeo, but base class can also do event-based corrections
# The dictionary keys correspond to specific corrections that can be toggled on (True) or off (False)
corrections = {
    "Pedestal": True,  # Correct for the baseline offset of the PMT signal
    "RateDepElec": True,  # Correct for the rate-dependent electronics artifacts
    "QDC": False,  # QDC non-linearity correction, generally not recommended
}

# The auto4corr() method is a higher-level function that automatically applies
# the specified corrections to the data. The set_mode parameter determines whether (0)
# default settings are applied, i.e., only filtering invalid events and cycles, (1) custom
# filters are used as set with set_filt(), and (2) if raw data is used without filters.
# Here, peds is a placeholder for the pedestal values which need to be determined
# from the data or set to a reasonable default, such as zeros.
data.auto4corr(set_mode=1, peds=np.zeros((16, 6)), corr_dict=corrections)

# After corrections, we may want to regenerate histograms to reflect the corrected data
data.gen_hist([])

# Plot the corrected histogram, specifying the range of interest. This range is specific
# to the type of data and the expected signal range, which needs to be determined based
# on the experiment's context.
data.hist_sums[0].plot_hist(rng=[0.0, 35e3, 0.0, 4e3])
