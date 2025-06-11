'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import glob

'''
# Find all matching CSV files
csv_files = sorted(glob.glob("BestBump_alphapt03_*_20250610.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by='slopeL', ascending=True)

# Save to a single merged file
df_all.to_csv("BestBump_alphapt03_20250610.csv", index=False)
print("csv merged")
'''

ramps = pd.read_csv("ramps2_grid_300_20250415.csv")
ramps["potential_H1"] = 0.300 + ramps["slopeH"]
ramps.to_csv("ramps2_grid_300_20250415.csv", index=False)