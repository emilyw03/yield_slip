'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import glob


# Find all matching CSV files
csv_files = sorted(glob.glob("BestBump_alphapt0_whole_*_20250611.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by='slopeL', ascending=True)

# Save to a single merged file
df_all.to_csv("BestBump_alpha0_whole_20250611.csv", index=False)
print("csv merged")


'''
df = pd.read_csv("BestBump_alphapt03_corner_20250605.csv")
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
df.to_csv("BestBump_alphapt03_corner_20250605.csv", index=False)
'''
