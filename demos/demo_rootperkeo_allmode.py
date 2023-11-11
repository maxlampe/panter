from panter.data.dataRootPerkeo import RootPerkeo

# Copied from the input parameters section of the RootPerkeo docstring
# >>> help(RootPerkeo)
"""
bfull_adc: False
        Store full ALLMODE data, too, instead of taking difference.
"""
# Import root file, but set bool bfull_adc to True
data = RootPerkeo("example.root", bfull_adc=True)

# Maximum number of ALLMODE events being used for memory optimization.
data.allmode_cut = 5000
# general note: panter does not use lazy arrays or similar structures.
# Could be potential for improvement, but I never ran into issues except ALLMODE data.

# Filter for only detector 0 has triggered.
data.set_filt(
    "data",
    fkey="CoinTime",
    active=True,
    ftype="num",
    low_lim=0,
    up_lim=4e9,
    index=0,
)
data.set_filt(
    "data",
    fkey="CoinTime",
    active=True,
    ftype="num",
    low_lim=4e9,
    up_lim=4e10,
    index=1,
)
data.auto(1)

# Copied from the attributes section of the RootPerkeo docstring
# >>> help(RootPerkeo)
"""
    pmt_data: np.array
        Arrays with all PMT data after generation with filters
    data_pmt_tran: np.array
        Arrays with all PMT data after generation with filters without taking the
        QDC integrator difference. Used for ALLMODE data.
"""
# Allmode data as np.array for further use.
# _tran refers to the data being transposed compared to the raw imported data
allmode_data_0 = data.data_pmt_tran
