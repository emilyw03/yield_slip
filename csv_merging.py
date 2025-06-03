'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import glob

# Find all matching CSV files
csv_files = sorted(glob.glob("2cof_float3_interval_*_20250602.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by='alpha', ascending=True)

# Save to a single merged file
df_all.to_csv("2cof_float3_interval_20250603.csv", index=False)
print("csv merged")
