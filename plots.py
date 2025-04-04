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
# gamma
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
# plot optimizer convergence
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


# 50alphas
df = pd.read_csv("DHLOpt_50alphas_gamma100_20250220.csv")

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

'''
# objective function F vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, F, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('F', fontsize=22)
plt.title(r'F vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()'''

'''
# plot dG vs. alpha for unrestricted dG
dG = -(pL2 + pH2)
plt.figure(figsize=(8, 5))
plt.plot(alphas, dG, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel(r'$\Delta$G', fontsize=22)
plt.title(r'$\Delta$G vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.show()'''

'''
ratio = fluxLR/fluxHR

# plot ratio vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas, ratio, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('FluxLR/FluxHR', fontsize=22)
plt.title(r'LR:HR Electron Partitioning vs. $\alpha$: $\Delta$G = -0.1 eV', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
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
plt.plot(alphas, pL3, label='L3', linestyle='-', color='red', linewidth = 3)
plt.plot(alphas, pH3, label='H3', linestyle='-', color='blue', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('Reduction potentials (eV)', fontsize=22)
plt.title(r'Optimized cofactor reduction potentials vs. $\alpha$: $\Delta$G = -0.1 eV', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 'best', fontsize = 16)
#plt.grid(True)
plt.show()'''

'''
df = pd.read_csv("DHLOpt_50alphas_gamma100_20250220.csv")

## in-set plots
alphas_sub = df["alpha"][10:41]
fluxD_sub = df["abs(fluxD)"][10:41]
fluxHR_sub = df["fluxHR"][10:41]
fluxLR_sub = df["fluxLR"][10:41]
pD1_sub = df["potential_D1"][10:41]
pD2_sub = -pD1_sub
pL2_sub = df["potential_L2"][10:41]
pH2_sub = df["potential_H2"][10:41]

# plot ratio vs. alpha
ratio = fluxLR_sub/fluxHR_sub

plt.figure(figsize=(8, 5))
plt.plot(alphas_sub, ratio, linewidth = 3, color = 'blue')
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('FluxLR/FluxHR', fontsize=22)
plt.title(r'LR:HR Electron Partitioning vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

# Plot of fluxes vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas_sub, fluxD_sub, label='|DR|', linestyle='-', color='green', linewidth = 3)
plt.plot(alphas_sub, fluxHR_sub, label='HR', linestyle='-', color='blue', linewidth = 3)
plt.plot(alphas_sub, fluxLR_sub, label='LR', linestyle='-', color='red', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel(r'Flux (s$^{-1}$)', fontsize=22)
plt.title(r'Fluxes at reservoirs (s$^{-1}$) vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

# plot potentials vs. alpha
plt.figure(figsize=(8, 5))
plt.plot(alphas_sub, pD1_sub, label=r'D/D$^{-}$', linestyle='-', color='purple', linewidth = 3)
plt.plot(alphas_sub, pD2_sub, label=r'D$^{-}$/D$^{=}$', linestyle='-', color='cyan', linewidth = 3)
plt.plot(alphas_sub, pL2_sub, label='L3', linestyle='-', color='red', linewidth = 3)
plt.plot(alphas_sub, pH2_sub, label='H3', linestyle='-', color='blue', linewidth = 3)
plt.xlabel(r'$\alpha$', fontsize=22)
plt.ylabel('Reduction potentials (eV)', fontsize=22)
plt.title(r'Optimized cofactor reduction potentials vs. $\alpha$', fontsize=22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 'best', fontsize = 16)
plt.show()'''
