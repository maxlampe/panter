"""Dump"""

import numpy as np
import pandas as pd

df = pd.DataFrame(columns=["x", "y", "z"])

df = df.append({"x": 0, "y": 0, "z": [0, 3]}, ignore_index=True)
df = df.append({"x": 1, "y": 3, "z": [4, 4]}, ignore_index=True)
df = df.append({"x": 2, "y": 5, "z": [2, 2]}, ignore_index=True)

print(df)

print(df["x"])
print(df["z"])

dfs = df["z"].apply(pd.Series)
print(dfs[0])
