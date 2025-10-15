'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import numpy as np
import glob


'''
# === merged parallelized files ===
# Find all matching CSV files
csv_files = sorted(glob.glob("Nfn1_varyH_ramp_fixDist_*_20250711.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by=['slopeH'], ascending=True)

# Save to a single merged file
df_all.to_csv("Nfn1_varyH_ramp_fixDist_20250711.csv", index=False)
print("csv merged")
'''


# === filter for bifurcating only fluxD < 0, fluxH, fluxL > 0
df = pd.read_csv("bump_Fyield_20250629.csv")
filtered = df[(df['fluxD'] < 0) & (df['fluxHR'] > 0) & (df['fluxLR'] > 0)]
#filtered = df[(df['NADPH_flux'] < 0) & (df['NAD_flux'] > 0) & (df['Fd_flux'] > 0)]
filtered.to_csv("bump_Fyield_bif_20250629.csv", index=False)

