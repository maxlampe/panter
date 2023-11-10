from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.corrPerkeo import CorrPerkeo

# Initialize data loader and load filtered measurements
data_dir = "/path/to/data/"
dataloader = DLPerkeo(data_dir)
dataloader.auto()
# Filter for beam data
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])[0]

# Initialize CorrPerkeo with the filtered measurements
# Mode sets type of calculated spectra.
#       O = total DetSum (only one spectra is returned)
#       1 = only Sum over each detector PMTs (two spectra are returned)
#       2 = all PMTs will be treated individually (no_pmt spectra)
corr_class = CorrPerkeo(filt_meas, mode=0)

# Set all corrections to False then selectively enable desired corrections
corr_class.set_all_corr(bactive=False)
corr_class.corrections["Drift"] = True
corr_class.corrections["Scan2D"] = True
corr_class.corrections["RateDepElec"] = True
corr_class.corrections["Pedestal"] = True
corr_class.corrections["DeadTime"] = True

# Add additional filters, if needed
corr_class.addition_filters.append(
    {"tree": "data", "fkey": "Detector", "active": True, "ftype": "bool", "rightval": 0}
)

# Perform corrections and store the corrected histograms
corr_class.corr(bstore=True, bwrite=False, bconcat=True)

# Access the histograms and plot them
histp = corr_class.histograms[0]
test = histp[1][0]
test.plot_hist(
    title="Beta spectrum",
    xlabel="Energy [ch]",
    ylabel="Counts [ ]",
    rng=[0.0, 50e3, -30.0, 3500.0],
)
print(test.stats)
