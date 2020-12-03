"""Dump"""

import ROOT
import pandas as pd


direct = "/mnt/sda/PerkeoDaten1920/cycle201/cycle201/"
filename2 = "data119886-67506_bg.root"
filename3 = "data119874-67506_2.root"

# import
histFileName = direct + filename3
dataTree = ROOT.RDataFrame("dataTree", histFileName)
cycleTree = ROOT.RDataFrame("cycleTree", histFileName)

# testing for DPTT generation
print("DeltaPrevTriggerTime" in dataTree.GetColumnNames())
print("DeltaTriggerTime" in dataTree.GetColumnNames())

det0PMT = dataTree

if True:
    # testing filter
    filter = ["Detector == 0", "PMT[0] > 3000", "DetSum > 4000", "DetSum < 10000"]
    # generate PMT data according to filters
    for filt in filter:
        det0PMT = det0PMT.Filter(filt)

# generate dataframe with n_events of cppyy.gbl.ROOT.VecOps.RVec<short>
# i.e. [n_events][n_pmt]
det0PMT = pd.DataFrame(det0PMT.AsNumpy()["PMT"])

print("Entire data result (with filter)")
print(f"Shape: {det0PMT.shape}")
print(f"Type: {type(det0PMT)}")
# print(f"Dir: {dir(det0PMT)}")
# print(det0PMT)

evno = [1, 2, 3, 4, det0PMT.size - 1]
for ev in evno:
    print(f"Event no {ev}")
    val = det0PMT.loc[ev]
    print(type(val))
    # print(dir(val))
    print(val)

exit()

# get all events for 1 PMT
n_pmt = 0
print(f"Only PMT{n_pmt}")
# print(det0PMT[0][n_pmt])
