'''
bumpy_vs_ramps.py
Author: Emily Wang
May 4, 2025
Comparing ranges of yield and slippage in bumpy vs. ramp landscapes
'''

import numpy as np
from numpy.linalg import eig

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
# === ramps ===
ramps = pd.read_csv('ramps2_grid_300_bif_20250415.csv')
fluxD = ramps["fluxD"]
fluxHR = ramps["fluxHR"]
fluxLR = ramps["fluxLR"]
ramps2 = ramps[(fluxD < 0 ) & (fluxHR > 0) & (fluxLR > 0) & (fluxLR <= fluxHR)]
ramps2.to_csv("ramps2_grid_300_bif_noback_20250415.csv", index=False)

# flux
rflux_min_idx = ramps['fluxD'].idxmax() # flipped since ramps data does not have abs value
rflux_max_idx = ramps['fluxD'].idxmin()
rflux_min_row = ramps.loc[rflux_min_idx, ['slopeL', 'slopeH', 'fluxD']]
rflux_max_row = ramps.loc[rflux_max_idx, ['slopeL', 'slopeH', 'fluxD']]
print(rflux_min_row)
print(rflux_max_row)

# efficiency
ratio = ramps2['fluxLR'] / ramps2['fluxHR']
rslip_min_idx = ratio.idxmin()
rslip_max_idx = ratio.idxmax()
rslip_min_row = ramps.loc[rslip_min_idx, ['slopeH', 'slopeL', 'fluxLR', 'fluxHR']]
rslip_max_row = ramps.loc[rslip_max_idx, ['slopeH', 'slopeL', 'fluxLR', 'fluxHR']]
print(rslip_min_row)
print(rslip_max_row)

# === bumpy ===
bumpy = pd.read_csv('2cof_float3_gamma1_all_20250421.csv')

# flux 
bflux_min_idx = bumpy['abs(fluxD)'].idxmin()
bflux_max_idx = bumpy['abs(fluxD)'].idxmax()
bflux_min_row = bumpy.loc[bflux_min_idx, ['abs(fluxD)', 'potential_D1', 'potential_H2', 'potential_L2']]
bflux_max_row = bumpy.loc[bflux_max_idx, ['abs(fluxD)', 'potential_D1', 'potential_H2', 'potential_L2']]
print(bflux_min_row)
print(bflux_max_row)

# efficiency
ratio = bumpy['fluxLR'] / bumpy['fluxHR']
bslip_min_idx = ratio.idxmin()
bslip_max_idx = ratio.idxmax()
bslip_min_row = bumpy.loc[bslip_min_idx, ['fluxLR', 'fluxHR', 'potential_D1', 'potential_H2', 'potential_L2']]
bslip_max_row = bumpy.loc[bslip_max_idx, ['fluxLR', 'fluxHR', 'potential_D1', 'potential_H2', 'potential_L2']]
print(bslip_min_row)
print(bslip_max_row)'''




