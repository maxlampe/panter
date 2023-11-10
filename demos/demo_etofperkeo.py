from panter.data.dataloaderPerkeo import DLPerkeo
from panter.eval.etofPerkeo import EToFPerkeo

# Assuming 'data_dir' is the directory containing the data files
data_dir = "/path/to/data/directory"
dataloader = DLPerkeo(data_dir)

# Automatically process the data directory to filter measurements
dataloader.auto()

# Retrieve filtered measurements based on specific criteria
# Example criteria: measurement type 'tp' and source type 'src' --> beam data
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [0, 5])[0]

# Initialize the EToFPerkeo class with the filtered measurements
# Set bplot_res to True to plot the resulting eToF histogram
etof = EToFPerkeo(filt_meas, bplot_res=True)

# The above instantiation automatically calculates the eToF
# The histogram is stored in a HistPerkeo object

# Plot the eToF histogram (if not automatically plotted during initialization)
etof.plot_etof()
