'''
csv_merging.py
June 2, 2025
Author: Emily Wang

Use for merging csv's after parallelized slurm tasks
'''

import pandas as pd
import numpy as np
import glob


# === merged parallelized files ===
# Find all matching CSV files
csv_files = sorted(glob.glob("bump_Fsc_*_20250630.csv"))

# Load and concatenate all CSVs
df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
df_all = df_all.sort_values(by=['slopeL', 'slopeH'], ascending=True)

# Save to a single merged file
df_all.to_csv("bump_Fsc_20250630.csv", index=False)
print("csv merged")


'''
# === add dG ===
df = pd.read_csv("BestBump_alpha1_corner_20250618.csv")
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
df.to_csv("BestBump_alpha1_corner_20250618.csv", index=False)
'''

'''
# === filter for bifurcating only (dG <= 0)
df = pd.read_csv("bump_Fyield_20250629.csv")
filtered = df[(df['fluxD'] < 0) & (df['fluxHR'] > 0) & (df['fluxLR'] > 0)]
#filtered = df[(df['NADPH_flux'] < 0) & (df['NAD_flux'] > 0) & (df['Fd_flux'] > 0)]
filtered.to_csv("bump_Fyield_bif_20250629.csv", index=False)
'''

'''
# find non-bifurcating in bifurcating range
df = df.copy()
filtered = filtered.copy()
df["slopeH_rounded"] = df["slopeH"].round(6)
filtered["slopeH_rounded"] = filtered["slopeH"].round(6)

# Filter: keep rows in filtered whose rounded slopeH is not in df
filtered_only = filtered[~filtered["slopeH_rounded"].isin(df["slopeH_rounded"])]

# Save to CSV
filtered_only.to_csv("Nfn1_vary_slopeH_bump_nonbif_test_20250618.csv", index=False)
'''
'''
df = pd.read_csv("Nfn1_varyH_ramp_bif_20250620.csv")
print('=== F_slip ===')
print(min(df['F_slip']), max(df['F_slip']))
print('=== F_yield ===')
print(min(df['F_yield']), max(df['F_yield']))'''

'''
df = pd.read_csv('ramps_FscFyield_20250629.csv')
df['F_sc_inv'] = 1 / df['F_sc']
df.to_csv('ramps_FscFyield_20250629.csv', index = False)
'''