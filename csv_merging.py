'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import glob


# === merged parallelized files ===
# Find all matching CSV files
csv_files = sorted(glob.glob("BestBump_alphapt03_whole_*_20250612.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by='slopeL', ascending=True)

# Save to a single merged file
df_all.to_csv("BestBump_alphapt03_whole_20250612.csv", index=False)
print("csv merged")

'''
# === add dG ===
df = pd.read_csv("BestBump_alpha1_corner_20250611.csv")
slopeL = df['slopeL']
slopeH = df['slopeH']
fluxLR = df['fluxLR']
fluxHR = df['fluxHR']
pLR = -0.3 + 2*slopeL
pHR = 0.3 + 2*slopeH
dG1 = -(pLR + pHR)
dG2 = -(2*pHR)
eff = fluxLR / fluxHR

dG = (eff * dG1) + ((1 - eff) * dG2)
df['dG'] = dG
df.to_csv("BestBump_alpha1_corner_20250611.csv", index=False)
'''
'''
# === filter for bifurcating only (dG <= 0)
df = pd.read_csv("ramps_corner_20250612.csv")
filtered = df[df['dG'] <= 0]
filtered.to_csv('ramps_corner_bif_20250612.csv')'''