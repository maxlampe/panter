from panter.data.dataHistPerkeo import HistPerkeo
import numpy as np

# Generate random data to simulate experimental results
data = np.random.normal(5, 2, 1000)  # normal distribution, mean=5, std=2

# Initialize the HistPerkeo object with data
histogram = HistPerkeo(data=data, bin_count=20, low_lim=0, up_lim=10)

# Calculate statistics and scale the histogram
histogram.scal(fac=2)  # Scale the histogram by a factor of 2

# Perform arithmetic with histograms
another_data = np.random.normal(7, 1.5, 800)  # another dataset
histogram2 = HistPerkeo(data=another_data, bin_count=20, low_lim=0, up_lim=10)
histogram.addhist(
    hist_p=histogram2, fac=-0.5
)  # Add another histogram to the existing one
histogram.divbyhist(hist_p=histogram2)  # Divide by another histogram
histogram.multbyhist(hist_p=histogram2)  # Multiply by another histogram

# Plotting the histogram
histogram.plot_hist(title="Final Histogram", xlabel="Value", ylabel="Frequency")

# Return histogram as numpy histogram format
numpy_histogram = histogram.ret_asnumpyhist()

# Writing the histogram to a ROOT file (requires ROOT package)
# histogram.write2root(histname="myHistogram", filename="histogram_output")
