'''
plots
scripts for slippage vs. yield plots
Feb 2025
Author: Emily Wang
'''

import numpy as np
from numpy.linalg import eig

import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.cm as cm

'''
# === gamma ===
df = pd.read_csv("gamma1.csv")

alphas = df["alpha"]
fluxD = df["abs(fluxD)"]
fluxHR = df["fluxHR"]
fluxLR = df["fluxLR"]

# Plot of fluxes vs. alpha
plt.figure(figsize=(8, 5))
plt.yscale("log")
plt.plot(alphas, fluxD, label='abs(Flux DR)', linestyle='-', color='green')
plt.plot(alphas, fluxHR, label='Flux HR', linestyle='-', color='blue')
plt.plot(alphas, fluxLR, label='Flux LR', linestyle='-', color='red')
plt.xlabel('alpha')
plt.ylabel('log(flux)')
plt.title('Fluxes vs. alpha (gamma = 1)')
plt.legend()
plt.grid(True)
plt.show()'''

'''
# === plot optimizer convergence ===
df = pd.read_csv("DHLOpt_runs_serial_20250220.csv")

F_t = df["F_(t)"]
t = df['t'] + 1
fluxD = df["abs(fluxD)_(t)"]
fluxHR = df["fluxHR_(t)"]
fluxLR = df["fluxLR_(t)"]

plt.figure(figsize=(8, 5))
plt.plot(t, F_t, linestyle='-', color='blue', linewidth = 3)
plt.xlabel('t (runs)', fontsize=22)
plt.ylabel('Minimum F of first t runs', fontsize=22)
plt.title('Convergence of optimized objective function value', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, fluxD, label='|DR|', linestyle='-', color='green')
plt.plot(t, fluxHR, label='HR', linestyle='-', color='blue')
plt.plot(t, fluxLR, label='LR', linestyle='-', color='red')
plt.xlabel('t (runs)', fontsize=16)
plt.ylabel(r'Minimum flux of first t runs (s$^{-1}$)', fontsize=16)
plt.title('Convergence of fluxes for optimized landscape', fontsize=16)
plt.legend(loc = 'best')
plt.grid(True)
plt.show()
'''

# === 50alphas ===

df = pd.read_csv("2cof_float3_interval_20250603.csv")

alphas = df["alpha"]
F = df['F_t']
fluxD = df["abs(fluxD)"]
fluxHR = df["fluxHR"]
fluxLR = df["fluxLR"]
pD1 = df["potential_D1"]
pD2 = -pD1
pL2 = df["potential_L2"]
pH2 = df["potential_H2"]

'''# fluxes vs. partitioning
ratio = fluxLR / fluxHR
plt.figure(figsize=(8, 6))
plt.plot(ratio, fluxD, label='|DR|', linestyle='-', color='green', linewidth = 3)
plt.plot(ratio, fluxHR, label='HR', linestyle='-', color='blue', linewidth = 3)
plt.plot(ratio, fluxLR, label='LR', linestyle='-', color='red', linewidth = 3)
plt.xlabel('fluxLR/fluxHR', fontsize=22)
plt.ylabel(r'Flux (s$^{-1}$)', fontsize=22)
plt.title(r'Fluxes at reservoirs (s$^{-1}$) vs. LR:HR partitioning', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.show()'''

'''
# fluxes vs. H2
plt.figure(figsize=(8, 6))
plt.xlim(max(pH2) + 0.01, min(pH2) - 0.01)  # reverse x-axis to be consistent with alpha plots
plt.plot(pH2, fluxD, label='|DR|', linestyle='-', color='green', linewidth = 3)
plt.plot(pH2, fluxHR, label='HR', linestyle='-', color='blue', linewidth = 3)
plt.plot(pH2, fluxLR, label='LR', linestyle='-', color='red', linewidth = 3)
plt.xlabel('potential on H2 (eV)', fontsize=22)
plt.ylabel(r'Flux (s$^{-1}$)', fontsize=22)
plt.title(r'Fluxes at reservoirs (s$^{-1}$) vs. potential on H2 (eV)', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

# objective function F vs. alpha
plt.figure(figsize=(8, 6))
plt.plot(alphas, F, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('F', fontsize=22)
plt.title(r'F vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

ratio = fluxLR/fluxHR

# plot ratio vs. alpha
plt.figure(figsize=(8, 6))
plt.plot(alphas, ratio, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('FluxLR/FluxHR', fontsize=22)
plt.title(r'LR:HR Electron Partitioning vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()'''
'''
# Plot of fluxes vs. alpha
plt.figure(figsize=(8, 6))
plt.plot(alphas, fluxD, label='|DR|', linestyle='-', color='green', linewidth = 3)
plt.plot(alphas, fluxHR, label='HR', linestyle='-', color='blue', linewidth = 3)
plt.plot(alphas, fluxLR, label='LR', linestyle='-', color='red', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel(r'Flux (s$^{-1}$)', fontsize=22)
plt.title(r'Fluxes at reservoirs (s$^{-1}$) vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
#plt.grid(True)
plt.show()
'''
'''
# plot potentials vs. alpha
plt.figure(figsize=(8, 6))
plt.plot(alphas, pD1, label=r'D/D$^{-}$', linestyle='-', color='purple', linewidth = 3)
plt.plot(alphas, pD2, label=r'D$^{-}$/D$^{=}$', linestyle='-', color='cyan', linewidth = 3)
plt.plot(alphas, pL2, label='L2', linestyle='-', color='red', linewidth = 3)
plt.plot(alphas, pH2, label='H2', linestyle='-', color='blue', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('Reduction potentials (eV)', fontsize=22)
plt.title(r'Optimized cofactor reduction potentials vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 'best', fontsize = 16)
#plt.grid(True)
plt.show()
'''

# === ramps ===
ramps_w = pd.read_csv("ramps2_grid_300_20250415.csv")

slopeL_w = ramps_w["slopeL"]
slopeH_w = ramps_w["slopeH"]
F_slip_w = ramps_w["F_slip"]
F_yield_w = ramps_w["F_yield"]

bump = pd.read_csv('BestBump_alpha1_20250609.csv')
slopeL_b = bump['slopeL']
slopeH_b = bump['slopeH']
F_slip_b = bump["F_slip"]
F_yield_b = bump["F_yield"]

'''
# F_yield vs. slopeH
plt.figure(figsize=(8, 5))
plt.plot(slopeH_z, F_yield_z, color = 'blue', linewidth=3)
plt.title(r'$F_{yield}$ vs. slopeH')
plt.xlabel('slopeH (eV/cofactor)')
plt.ylabel(r'$F_{yield}$ (s)')
plt.show()

# F_slip vs. slopeH
plt.figure(figsize=(8, 5))
plt.plot(slopeH_z, F_slip_z, color = 'blue', linewidth=3)
plt.title(r'$F_{slip}$ vs. slopeH')
plt.xlabel('slopeH (eV/cofactor)')
plt.ylabel(r'$F_{slip}$')
plt.show()
'''
'''
# color by F_slip
vmin = min(np.min(F_slip_w), np.min(F_yield_w))
vmax = max(np.max(F_slip_w), np.max(F_yield_w))
norm = LogNorm(vmin=vmin, vmax=vmax)

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    slopeL_w, slopeH_w, c=F_slip_w, cmap='viridis',
    s=60, edgecolor='none', norm=norm
)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{F}_{\mathrm{slip}}$', fontsize=12)

# overlay bump points
plt.scatter(slopeL_b, slopeH_b, c=F_slip_b, cmap='viridis', norm=norm, marker='o', edgecolor='black', linewidths=0.5)

# labels
plt.suptitle(r'$\mathrm{F}_{\mathrm{slip}}$ by ET branch slopes')
plt.title(r'Overlay bump optimization for $\alpha=1$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()

# Base grid plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    slopeL_w, slopeH_w, c=F_yield_w, cmap='viridis',
    s=60, edgecolor='none', norm=norm
)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\mathrm{F}_{\mathrm{yield}}$', fontsize=12)

# overlay bump points
plt.scatter(slopeL_b, slopeH_b, c=F_yield_b, cmap='viridis', norm=norm, marker='o', edgecolor='black', linewidths=0.5)

# labels
plt.suptitle(r'$\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes')
plt.title(r'Overlay bump optimization for $\alpha=1$', fontsize=10)
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.tight_layout()
plt.legend()
plt.show()
'''
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
plt.show()
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