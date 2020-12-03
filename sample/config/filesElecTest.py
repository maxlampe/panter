#!/usr/bin/python3
"""General ElecTest measurement file information"""

ELECTEST_DIR = "/data/PerkeoDaten1920/ElecTest_20200309/"
# """ Test 1 """
TEST1_DIR = ELECTEST_DIR + "Test1_Testing_general_non-linearity/"

Test1_x1 = [30, 50, 100, 200, 400, 600, 800]  # [mV]
# BOTH mode data
# PMT0-3
Test1_smp1 = [
    "data236701.root",
    "data236713.root",
    "data236191.root",
    "data236161.root",
    "data236131.root",
    "data236101.root",
    "data236071.root",
]
# PMT4-7
Test1_smp2 = [
    "data236461.root",
    "data236431.root",
    "data236401.root",
    "data236371.root",
    "data236341.root",
    "data236311.root",
    "data236281.root",
]
# PMT8-11
Test1_smp3 = [
    "data236491.root",
    "data236521.root",
    "data236551.root",
    "data236569.root",
    "data236581.root",
    "data236593.root",
    "data236605.root",
]
# PMT12-15
Test1_smp4 = [
    "data236689.root",
    "data236677.root",
    "data236665.root",
    "data236653.root",
    "data236641.root",
    "data236629.root",
    "data236617.root",
]
# DELTA mode data
# PMT0-3
Test1_smp5 = [
    "data236725.root",
    "data236737.root",
    "data236749.root",
    "data236761.root",
    "data236773.root",
    "data236785.root",
    "data236797.root",
]
# PMT4-7
Test1_smp6 = [
    "data236881.root",
    "data236869.root",
    "data236857.root",
    "data236845.root",
    "data236833.root",
    "data236821.root",
    "data236809.root",
]
# PMT8-11
Test1_smp7 = [
    "data236893.root",
    "data236905.root",
    "data236917.root",
    "data236929.root",
    "data236941.root",
    "data236953.root",
    "data236965.root",
]
# PMT12-15
Test1_smp8 = [
    "data237049.root",
    "data237037.root",
    "data237025.root",
    "data237013.root",
    "data237001.root",
    "data236989.root",
    "data236977.root",
]
# only LeCroy FanOut
Test1_x2 = [20, 30, 50, 100, 200, 400, 600, 800]  # [mV]
# PMT0 and 3
Test1_smp9 = [
    "data237061.root",
    "data237073.root",
    "data237085.root",
    "data237097.root",
    "data237109.root",
    "data237121.root",
    "data237133.root",
    "data237145.root",
]


# """ Test 2 """

TEST2_DIR = ELECTEST_DIR + "Test2_Test_sweep_mode/"

Test2_x1 = [300, 200, 100, 50]
# burst mode 20ns pulse width
Test2_smp1 = [
    "data237181.root",
    "data237187.root",
    "data237205.root",
    "data237211.root",
]
# sweep mode 20ns pulse width
Test2_smp2 = [
    "data237175.root",
    "data237193.root",
    "data237199.root",
    "data237217.root",
]

Test2_x2 = [50, 300, 600]
# burst mode 10ns pulse width
Test2_smp3 = ["data237229.root", "data237241.root", "data237247.root"]
# sweep mode 10ns pulse width
Test2_smp4 = ["data237223.root", "data237235.root", "data237253.root"]

# """ Test 3 """

TEST3_DIR = ELECTEST_DIR + "Test3_Testing_sweep_mode_for_two_floating_outputs/"
# just import all. its only 4 files

# """ Test 4 """

TEST4_DIR = ELECTEST_DIR + "Test4_Using_double_sweep_mode_same_Ampl/"

Test4_x1 = [200, 300, 500]

Test4_0003_200 = [
    "data237433.root",
    "data237613.root",
    "data237793.root",
    "data237973.root",
]
Test4_0003_300 = [
    "data238153.root",
    "data238333.root",
    "data238513.root",
    "data238693.root",
]
Test4_0003_500 = ["data242653.root", "data242833.root"]

Test4_1215_200 = [
    "data245773.root",
    "data245953.root",
    "data246133.root",
    "data246313.root",
]
Test4_1215_300 = ["data250453.root", "data250633.root"]
Test4_1215_500 = ["data250813.root", "data250993.root"]

Test4_0811_200 = ["data251893.root", "data252073.root"]
Test4_0811_300 = ["data251533.root", "data251713.root"]
Test4_0811_500 = ["data251173.root", "data251353.root"]

Test4_0407_200 = ["data252253.root", "data252433.root"]
Test4_0407_300 = ["data252613.root", "data252793.root"]
Test4_0407_500 = ["data252973.root", "data253153.root"]

# """ Test 5 """

TEST5_DIR = ELECTEST_DIR + "Test5_Using_double_sweep_mode_diff_Ampl/"

Test5_0003_200500 = ["data243013.root", "data243193.root"]
Test5_0003_050300 = ["data243373.root", "data243553.root"]
Test5_0407_200500 = ["data244273.root", "data244093.root"]
Test5_0407_050300 = ["data243733.root", "data243913.root"]
Test5_0811_200500 = ["data244453.root", "data244633.root"]
Test5_0811_050300 = ["data244813.root", "data244993.root"]
Test5_1215_200500 = [
    "data245713.root",
    "data245653.root",
    "data245593.root",
    "data245533.root",
]
Test5_1215_050300 = ["data245173.root", "data245353.root"]
