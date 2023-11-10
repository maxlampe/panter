import numpy as np
from panter.data.dataHist2DPerkeo import Hist2DPerkeo

# Simulate some 2D data: two related distributions
x_data = np.random.normal(5, 1, 1000)
y_data = x_data + np.random.normal(0, 0.5, 1000)
data = np.array([x_data, y_data])

# Create a 2D histogram object with the data
hist_2d = Hist2DPerkeo(data=data, bin_count=[50, 50], low_lim=[0, 0], up_lim=[10, 10])

# Plot the 2D histogram with specified ranges and labels
hist_2d.plot_hist(
    rng=[0, 10, 0, 10],
    title="2D Histogram",
    xlabel="X-axis",
    ylabel="Y-axis",
    zlabel="Counts",
)

# If there's another dataset to compare with, add it to the current histogram
# Simulate another dataset for this example
additional_x_data = np.random.normal(4, 1, 1000)
additional_y_data = additional_x_data + np.random.normal(0, 0.5, 1000)
additional_data = np.array([additional_x_data, additional_y_data])
additional_hist = Hist2DPerkeo(
    data=additional_data, bin_count=[50, 50], low_lim=[0, 0], up_lim=[10, 10]
)

# Scale the additional histogram if needed (for example, to normalize)
additional_hist.scal(0.8)

# Add the scaled additional histogram to the original one
hist_2d.addhist(hist_p=additional_hist)

# Plot the combined 2D histogram
hist_2d.plot_hist(title="Combined 2D Histogram")

# Save the plot if needed
# hist_2d.plot_hist(bsavefig=True, filename="combined_2d_histogram")
