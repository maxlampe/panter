from panter.data.dataRootPerkeo import RootPerkeo
from panter.eval.pedPerkeo import PedPerkeo

# Load PERKEO III data from a ROOT file
data = RootPerkeo("path_to_data_file.root")

# Initialize PedPerkeo with the data
# Set bplot_fit and bplot_log to True to plot each fit result and use a log scale for the y-axis
ped_calc = PedPerkeo(dataclass=data, bplot_res=True, bplot_fit=True, bplot_log=True)

# Calculate pedestals for the PMTs, which will automatically plot the results if bplot_res is True
pedestals = ped_calc.calc_ped()

# Retrieve pedestal values
ped_values = ped_calc.ret_pedestals()

# Plotting pedestal fit results
ped_calc.plot_pedestals()

# Print the pedestal values
print(ped_values)
