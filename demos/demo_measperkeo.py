from panter.data.dataMeasPerkeo import MeasPerkeo
from panter.data.dataloaderPerkeo import DLPerkeo

# Assume 'file_list' contains paths to the measurement files
file_list = ["path/to/measurement1.dat", "path/to/measurement2.dat"]

# Initialize a MeasPerkeo object
# file_list: List of file(s) for a single measurement. One file for tp = 0 and 2, two for tp = 1.
# tp: States whether beam-like (0), source-like data(1) or without background(2), for background subtraction.
# src: Measurment type. Convention: 0-4 calibration sources, 5 beam, 6-7 background
# w/wo B field and 8 would be electronic tests.
meas = MeasPerkeo(tp=1, src=2, file_list=file_list)

# Print the MeasPerkeo object to check its string representation
print(meas)

# Initialize the data loader with path to a data directory
# DLPerkeo automatically generates arrays of MeasPerkeo objects based on file names
dir = "path-to-data"
dataloader = DLPerkeo(dir)
dataloader.auto()

# You can also initialize on an empty path and manually add events
dataloader = DLPerkeo("")
events = [
    MeasPerkeo(0, 5, list(["dir/data11-22_beam.root"]), 11, 22),
    MeasPerkeo(0, 7, list(["dir/data.root"])),
]
dataloader.fill(events)

# Get all measurements returned as array
all_meas = dataloader.ret_meas()
# Or apply a filter based on MeasPerkeo attributs
filt_meas = dataloader.ret_filt_meas(["tp", "src"], [1, 2])


# Perform further processing or analysis on all_meas or filt_meas
