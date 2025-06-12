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
import seaborn as sns


# === ramps ===

ramps_w = pd.read_csv("ramps_whole_20250612.csv")
F_slip_w = ramps_w["F_slip"]
F_yield_w = ramps_w["F_yield"]

ramps = pd.read_csv("ramps_whole_20250612.csv")
slopeL_r = ramps["slopeL"]
slopeH_r = ramps["slopeH"]
F_slip_r = ramps["F_slip"]
F_yield_r = ramps["F_yield"]
#dG_r = ramps['dG']

bump = pd.read_csv('BestBump_alphapt03_whole_20250612.csv')
slopeL_b = bump['slopeL']
slopeH_b = bump['slopeH']
F_slip_b = bump["F_slip"]
F_yield_b = bump["F_yield"]
#dG_b = bump['dG']'''

'''
# === color by dG === 
#all_dG = pd.concat([dG_r, dG_b])
#vmin = -all_dG.max()
#vmax = -all_dG.min()

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL_r, slopeH_r, c=-dG_r, cmap='viridis', s=60, edgecolor='none', vmin=-0.5, vmax=0.5)
cbar = plt.colorbar(sc)
cbar.set_label(r'Energy loss (-$\Delta\mathrm{G}$)', fontsize=12)

# overlay bump points
#plt.scatter(slopeL_b, slopeH_b, c=-dG_b, cmap='viridis', marker='o', edgecolor='black', linewidths=0.5, vmin=-0.5, vmax=0.5)

# labels
plt.suptitle(r'Energy loss (-$\Delta\mathrm{G}$) by ET branch slopes')
plt.title(r'Overlay bump optimization for $\alpha=1$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()
'''

# === color by F_slip and F_yield ===
# color by F_slip
# color bar is based on the ranges of F_slip and F_yield for ramps whole square
vmin = min(np.min(F_slip_w), np.min(F_yield_w))
vmax = max(np.max(F_slip_w), np.max(F_yield_w))
norm = LogNorm(vmin=vmin, vmax=vmax)

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(slopeL_r, slopeH_r, c=F_slip_r, cmap='viridis', s=60, edgecolor='none', norm=norm)
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
sc = plt.scatter(slopeL_r, slopeH_r, c=F_yield_r, cmap='viridis', s=60, edgecolor='none', norm=norm)
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


# === bump vs ramp plots ===
'''
ramps = pd.read_csv("ramps_corner_bif_20250612.csv")
bump = pd.read_csv("BestBump_alpha1_corner_bif_20250611.csv")
ramps_coords = ramps[["slopeL", "slopeH"]].to_numpy()
bump_coords = bump[["slopeL", "slopeH"]].to_numpy()

pH1_disp = bump['potential_H1'] - ramps['potential_H1']

# nearest neighbor matching (since grid searches aren't the same size)
tree = cKDTree(ramps_coords)
dists, indices = tree.query(bump_coords)

pH1_b = bump['potential_H1'].to_numpy()
pH1_r = ramps['potential_H1'].to_numpy()[indices]
pH1_disp = pH1_b - pH1_r

F_slip_b = bump["F_slip"].to_numpy()
F_slip_r = ramps["F_slip"].to_numpy()[indices]
F_slip_diff = F_slip_b / F_slip_r


# bin data
df = pd.DataFrame({"pH1_disp": pH1_disp, "F_slip_diff": F_slip_diff})
N = 20
df["bin"] = pd.cut(df["pH1_disp"], bins=N)

# Group by bin and compute statistics
grouped = df.groupby("bin").agg(
    mean_disp=("pH1_disp", "mean"),
    mean_diff=("F_slip_diff", "mean"),
    std_diff=("F_slip_diff", "std")
).dropna()

# Plot with error bars
plt.errorbar(
    grouped["mean_disp"], grouped["mean_diff"],
    yerr=grouped["std_diff"], fmt='o', capsize=4, color='blue'
)

plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
plt.text(x=grouped["mean_disp"].max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
plt.yscale('log')
plt.xlabel('H1 displacement (bump - ramp) (eV)')
plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ (bump/ramp)')
plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. H1 Displacement (trend)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(pH1_disp, F_slip_diff, color='blue')
plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. H1 Displacement (all points)')
plt.yscale('log')
plt.xlabel('H1 displacement (bump - ramp) (eV)')
plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ (bump/ramp)')
plt.tight_layout()
plt.show()'''

'''
plt.figure(figsize=(8, 6))
plt.scatter(pH1_disp, F_yield_diff, color='blue')
plt.yscale('log')
plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$ vs. H1 Displacement')
plt.title(r'(bump, $\alpha=1$)/(ramp)', fontsize=10)
plt.xlabel('H1 displacement (bump - ramp) (eV)')
plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$')
plt.tight_layout()
plt.show()
'''

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