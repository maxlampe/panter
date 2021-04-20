"""Creating a 2D map of the from the scan results from the 21st January, 2020"""

from panter.core.dataloaderPerkeo import DLPerkeo
from panter.core.corrPerkeo import CorrPerkeo

positions = [
    [175, 5348],
    [175, 5770],
    [175, 5920],
    [175, 6070],
    [175, 6370],
    [175, 5346],
    [175, 5620],
    [175, 5470],
    [175, 5320],
    [175, 5170],
    [175, 5020],
    [170, 5770],
    [170, 5920],
    [170, 5620],
    [177, 5348],
    [170, 6070],
    [170, 5470],
    [170, 6370],
    [170, 5170],
    [160, 5770],
    [160, 6070],
    [160, 6470],
    [175, 5349],
    [180, 5770],
    [180, 6070],
    [180, 5470],
]

tar_dir = "/mnt/sda/PerkeoDaten1920/cycle201/Det200121"
dataloader = DLPerkeo(tar_dir)

# format [tp, src, file_list, cycle_no]
#    tp=1 type of measurement (bg in separate file)
#    src=3 which source (Sn calibration source)
#    cycle_no (i.e. unique measurement ID)

events = [
    [1, 3, list([f"{tar_dir}/data189554.root", f"{tar_dir}/data189550.root"]), 189554],
    [1, 3, list([f"{tar_dir}/data189560.root", f"{tar_dir}/data189550.root"]), 189560],
    [1, 3, list([f"{tar_dir}/data189566.root", f"{tar_dir}/data189550.root"]), 189566],
    [1, 3, list([f"{tar_dir}/data189572.root", f"{tar_dir}/data189584.root"]), 189572],
    [1, 3, list([f"{tar_dir}/data189578.root", f"{tar_dir}/data189584.root"]), 189578],
    [1, 3, list([f"{tar_dir}/data189588.root", f"{tar_dir}/data189584.root"]), 189588],
    [1, 3, list([f"{tar_dir}/data189594.root", f"{tar_dir}/data189584.root"]), 189594],
    [1, 3, list([f"{tar_dir}/data189600.root", f"{tar_dir}/data189584.root"]), 189600],
    [1, 3, list([f"{tar_dir}/data189606.root", f"{tar_dir}/data189584.root"]), 189606],
    [1, 3, list([f"{tar_dir}/data189612.root", f"{tar_dir}/data189624.root"]), 189612],
    [1, 3, list([f"{tar_dir}/data189618.root", f"{tar_dir}/data189624.root"]), 189618],
    [1, 3, list([f"{tar_dir}/data189628.root", f"{tar_dir}/data189624.root"]), 189628],
    [1, 3, list([f"{tar_dir}/data189634.root", f"{tar_dir}/data189624.root"]), 189634],
    [1, 3, list([f"{tar_dir}/data189640.root", f"{tar_dir}/data189652.root"]), 189640],
    [1, 3, list([f"{tar_dir}/data189646.root", f"{tar_dir}/data189652.root"]), 189646],
    [1, 3, list([f"{tar_dir}/data189656.root", f"{tar_dir}/data189652.root"]), 189656],
    [1, 3, list([f"{tar_dir}/data189662.root", f"{tar_dir}/data189652.root"]), 189662],
    [1, 3, list([f"{tar_dir}/data189668.root", f"{tar_dir}/data189680.root"]), 189668],
    [1, 3, list([f"{tar_dir}/data189674.root", f"{tar_dir}/data189680.root"]), 189674],
    [1, 3, list([f"{tar_dir}/data189684.root", f"{tar_dir}/data189680.root"]), 189684],
    [1, 3, list([f"{tar_dir}/data189690.root", f"{tar_dir}/data189680.root"]), 189690],
    [1, 3, list([f"{tar_dir}/data189696.root", f"{tar_dir}/data189702.root"]), 189696],
    [1, 3, list([f"{tar_dir}/data189706.root", f"{tar_dir}/data189702.root"]), 189706],
    [1, 3, list([f"{tar_dir}/data189712.root", f"{tar_dir}/data189702.root"]), 189712],
    [1, 3, list([f"{tar_dir}/data189718.root", f"{tar_dir}/data189702.root"]), 189718],
    [1, 3, list([f"{tar_dir}/data189724.root", f"{tar_dir}/data189730.root"]), 189724],
]

dataloader.fill(events)

assert len(positions) == len(dataloader), "ERROR: #Positions != #Measurements"

meas = dataloader.ret_meas()
corr_class = CorrPerkeo(meas, mode=1)
corr_class.set_all_corr(bactive=True)
corr_class.corrections["Drift"] = False
corr_class.corr(bstore=True, bwrite=False)

for ind, hist in corr_class.histograms:
    print(positions[ind])
    hist[1].plot_hist()
    # Instead of plotting, you can us the HistPerkeo object as input for fitting
    # Hint: evalPerkeo.DoFit would be your next move.
