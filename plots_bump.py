'''
plots_bump.py
Author: Emily Wang
Date: June 11, 2025
plots for assessing the advantage of a bump in HPB vs. ramps
'''
import numpy as np
from numpy.linalg import eig

import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.cm as cm

# === ramps ===
ramps_w = pd.read_csv("ramps2_grid_300_20250415.csv")
F_slip_w = ramps_w["F_slip"]
F_yield_w = ramps_w["F_yield"]

ramps = pd.read_csv("ramps2_grid_300_corner_20250504.csv")
slopeL_r = ramps["slopeL"]
slopeH_r = ramps["slopeH"]
F_slip_r = ramps["F_slip"]
F_yield_r = ramps["F_yield"]

bump = pd.read_csv('BestBump_alpha1_corner_20250610.csv')
slopeL_b = bump['slopeL']
slopeH_b = bump['slopeH']
F_slip_b = bump["F_slip"]
F_yield_b = bump["F_yield"]

# === color by F_slip and F_yield ===
# color by F_slip
# color bar is based on the ranges of F_slip and F_yield for ramps whole square
vmin = min(np.min(F_slip_w), np.min(F_yield_w))
vmax = max(np.max(F_slip_w), np.max(F_yield_w))
norm = LogNorm(vmin=vmin, vmax=vmax)

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    slopeL_r, slopeH_r, c=F_slip_r, cmap='viridis',
    s=60, edgecolor='none', norm=norm
)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{F}_{\mathrm{slip}}$', fontsize=12)

# overlay bump points
plt.scatter(slopeL_b, slopeH_b, c=F_slip_b, cmap='viridis', norm=norm, marker='o', edgecolor='black', linewidths=0.5)

# labels
plt.suptitle(r'$\mathrm{F}_{\mathrm{slip}}$ by ET branch slopes')
plt.title(r'Overlay bump optimization for $\alpha=0.03$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    slopeL_r, slopeH_r, c=F_yield_r, cmap='viridis',
    s=60, edgecolor='none', norm=norm
)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{F}_{\mathrm{yield}}$', fontsize=12)

# overlay bump points
plt.scatter(slopeL_b, slopeH_b, c=F_yield_b, cmap='viridis', norm=norm, marker='o', edgecolor='black', linewidths=0.5)

# labels
plt.suptitle(r'$\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes')
plt.title(r'Overlay bump optimization for $\alpha=0.03$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()

'''
# color by fluxH
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL, slopeH, c=fluxHR, cmap='viridis', s=60, edgecolor='none')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{Flux}_{\mathrm{HR}}$', fontsize=12)
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
plt.title(r'$\mathrm{Flux}_{\mathrm{HR}}$ by ET branch slopes')
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.show()

# color by fluxL
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL, slopeH, c=fluxLR, cmap='viridis', s=60, edgecolor = 'none')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{Flux}_{\mathrm{LR}}$', fontsize=12)
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
plt.title(r'$\mathrm{Flux}_{\mathrm{LR}}$ by ET branch slopes')
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.show()'''

# === additional bump optimization plots ===
# pH1 vs. slopeH w/ slopeL held constant
'''
bump_alpha0 = pd.read_csv('BestBump_alpha0_20250606.csv')
filtered = bump_alpha0[bump_alpha0['slopeL'] == 0]
slopeL = filtered['slopeL']
slopeH = filtered['slopeH']
pH1 = filtered['potential_H1']
pH1_ramp = filtered['potential_H1_ramp']

plt.figure(figsize=(7, 6))
plt.scatter(slopeH, pH1, color='blue', label='bump')
plt.scatter(slopeH, pH1_ramp, color='red', label='ramp')
plt.title('Potential on H1 vs. HPB Slope (slopeL = 0 eV/cofactor)')
plt.xlabel('slopeH (eV/cofactor)')
plt.ylabel('Potential on H1 (eV)')
plt.tight_layout()
plt.legend()
plt.show()
'''
'''
bump_0 = pd.read_csv('BestBump_alpha0_20250606.csv')
bump_1 = pd.read_csv('BestBump_alpha1_20250609.csv')

'''
'''
# === difference plots ===
# both datasets have the same slope grid search combinations, so slopes from either bump_0 or bump_1 work
slopeL = bump_0['slopeL']
slopeH = bump_0['slopeH']
F_slip_diff = bump_1['F_slip'] / bump_0['F_slip']
F_yield_diff = bump_1['F_yield'] / bump_0['F_yield']

# F_slip diff
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL, slopeH, c=F_slip_diff, cmap='viridis', s=60, edgecolor='none', vmin = 0, vmax = 1)
cbar = plt.colorbar(sc)
cbar.set_label(r'Relative $\Delta \mathrm{F}_{\mathrm{slip}}$', fontsize=12)

# labels
plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ by ET branch slopes')
plt.title(r'$(\alpha=1) / (\alpha=0)$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()

# F_yield diff
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL, slopeH, c=F_yield_diff, cmap='viridis', s=60, edgecolor='none', vmin = 1, vmax = 2)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\Delta \mathrm{F}_{\mathrm{yield}}$', fontsize=12)

# labels
plt.suptitle(r'$\Delta\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes')
plt.title(r'$(\alpha=1) / (\alpha=0)$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()'''

'''
# === bump vs ramp plots ===
ramps_w = pd.read_csv("ramps2_grid_300_20250415.csv")
ramps_coords = ramps_w[["slopeL", "slopeH"]].to_numpy()
bump_coords = bump_1[["slopeL", "slopeH"]].to_numpy() # starting with alpha = 1 (check other alphas later)

pH1 = bump_1['potential_H1']

# nearest neighbor matching (since grid searches aren't the same size)
tree = cKDTree(ramps_coords)
dists, indices = tree.query(bump_coords)
F_slip_b = bump_1["F_slip"].to_numpy()
F_slip_r_matched = ramps_w["F_slip"].to_numpy()[indices]
F_yield_b = bump_1["F_yield"].to_numpy()
F_yield_r_matched = ramps_w["F_yield"].to_numpy()[indices]

F_slip_diff = F_slip_b / F_slip_r_matched
F_yield_diff = F_yield_b / F_yield_r_matched

plt.figure(figsize=(8, 6))
plt.scatter(pH1, F_slip_diff, color='blue', s=60)
plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. H1 Potential')
plt.title(r'(bump, $\alpha=1$)/(ramp)', fontsize=10)
plt.xlabel('H1 potential (eV)')
plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$')
plt.tight_layout()
plt.show()'''

'''
# === ramps BayOpt search === 
df = pd.read_csv('ramps2_iters_20250419.csv')

slopeL = df['slopeL']
slopeH = df['slopeH']

colors = np.arange(len(df))
plt.figure(figsize=(8, 5))
scatter = plt.scatter(slopeL, slopeH, c=colors, cmap='viridis', edgecolor='none')

# Add colorbar
plt.colorbar(scatter, label='Iteration')

# Label axes
plt.xlabel('slopeL')
plt.ylabel('slopeH')
plt.title('Slope pairs searched in one Bayesian Optimization run')
plt.show()'''