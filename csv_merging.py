'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import glob

# Find all matching CSV files
csv_files = sorted(glob.glob("BestBump_alpha1_*_20250609.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by='slopesL', ascending=True)
df_all = df_all.rename(columns={
    'slopesL': 'slopeL',
    'F_t': 'F',
    'F_slip_best': 'F_slip',
    'F_yield_best': 'F_yield'
})

# Save to a single merged file
df_all.to_csv("BestBump_alpha1_20250609.csv", index=False)
print("csv merged")
