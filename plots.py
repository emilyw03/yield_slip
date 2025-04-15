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
'''
# === 50alphas ===
df = pd.read_csv("2cof_float3_20250401.csv")

alphas = df["alpha"]
F = df['F_t']
fluxD = df["abs(fluxD)"]
fluxHR = df["fluxHR"]
fluxLR = df["fluxLR"]
pD1 = df["potential_D1"]
pD2 = -pD1
pL2 = df["potential_L2"]
pH2 = df["potential_H2"]

# fluxes vs. H2
plt.figure(figsize=(8, 5))
plt.yscale("log")
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
plt.figure(figsize=(8, 5))
plt.plot(alphas, F, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('F', fontsize=22)
plt.title(r'F vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()


# plot dG vs. alpha for unrestricted dG
dG = -(pL2 + pH2)
plt.figure(figsize=(8, 5))
plt.plot(alphas, dG, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel(r'$\Delta$G', fontsize=22)
plt.title(r'$\Delta$G vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

ratio = fluxLR/fluxHR

# plot ratio vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, ratio, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('FluxLR/FluxHR', fontsize=22)
plt.title(r'LR:HR Electron Partitioning vs. $\alpha$: $\Delta$G = -0.1 eV', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

# Plot of fluxes vs. alpha
plt.figure(figsize=(8, 5))
plt.yscale("log")
plt.plot(alphas, fluxD, label='|DR|', linestyle='-', color='green', linewidth = 3)
plt.plot(alphas, fluxHR, label='HR', linestyle='-', color='blue', linewidth = 3)
plt.plot(alphas, fluxLR, label='LR', linestyle='-', color='red', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel(r'Flux (s$^{-1}$)', fontsize=22)
plt.title(r'Fluxes at reservoirs (s$^{-1}$) vs. $\alpha$: $\Delta$G = -0.1 eV', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
#plt.grid(True)
plt.show()

# plot potentials vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, pD1, label=r'D/D$^{-}$', linestyle='-', color='purple', linewidth = 3)
plt.plot(alphas, pD2, label=r'D$^{-}$/D$^{=}$', linestyle='-', color='cyan', linewidth = 3)
plt.plot(alphas, pL2, label='L2', linestyle='-', color='red', linewidth = 3)
plt.plot(alphas, pH2, label='H2', linestyle='-', color='blue', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('Reduction potentials (eV)', fontsize=22)
plt.title(r'Optimized cofactor reduction potentials vs. $\alpha$: $\Delta$G = -0.1 eV', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 'best', fontsize = 16)
#plt.grid(True)
plt.show()'''

# === ramps ===
df = pd.read_csv("ramps2_grid_20250415.csv")

slopeL = df["slopeL"]
slopeH = df["slopeH"]
F_slip = df["F_slip"]
F_yield = df["F_yield"]
fluxD = df["fluxD"]
fluxHR = df["fluxHR"]
fluxLR = df["fluxLR"]

df_bif = df[(fluxD < 0 ) & (fluxHR > 0) & (fluxLR > 0)]
x_min, x_max = slopeL.min() - 0.01, slopeL.max() + 0.01
y_min, y_max = slopeH.min() - 0.01, slopeH.max() + 0.01

# color by F_slip
plt.figure(figsize=(8, 5))
sc = plt.scatter(df_bif['slopeL'], df_bif['slopeH'], c=df_bif['F_slip'], cmap='viridis', s=60, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label(r'$F_{slip}$', fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(r'$F_{slip}$ by ET branch slopes (bifurcating only)')
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.show()

# color by F_yield
plt.figure(figsize=(8, 5))
sc = plt.scatter(df_bif['slopeL'], df_bif['slopeH'], c=df_bif['F_yield'], cmap='viridis', s=60, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label(r'$F_{yield}$', fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(r'$F_{yield}$ by ET branch slopes (bifurcating only)')
plt.xlabel('slopeL (eV/cofactor)')
plt.ylabel('slopeH (eV/cofactor)')
plt.show()
