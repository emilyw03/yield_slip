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

def grid_Fslip(F_slip_w, F_yield_w, slopeL_r, slopeH_r, slopeL_b, slopeH_b, F_slip_r, F_slip_b):
    '''
    plot grid search with bump points overlayed on ramps, color by F_slip
    Arguments:
        F_slip_w (vector): Vector of F_slip data for ramps, whole square
        F_yield_w (vector): Vector of F_yield data for ramps, whole square
        slopeL_* (vector): Vector of slopeL data for ramps or bumps
        slopeH_* (vector): Vector of slopeH data for ramps or bumps
        F_slip_* (vector): Vector of F_slip data for ramps or bumps
    '''

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
    plt.scatter(slopeL_b, slopeH_b, c=F_slip_b, cmap='viridis', s=20, norm=norm, marker='o', edgecolor='black', linewidths=0.5)

    # labels
    plt.suptitle(r'$\mathrm{F}_{\mathrm{slip}}$ by ET branch slopes')
    plt.title(r'Overlay bump optimization for $\alpha=1$', fontsize=10)
    plt.xlabel('slopeL (eV/cofactor)')
    plt.ylabel('slopeH (eV/cofactor)')
    plt.tight_layout()
    plt.legend()
    plt.show()

def grid_Fyield(F_slip_w, F_yield_w, slopeL_r, slopeH_r, slopeL_b, slopeH_b, F_yield_r, F_yield_b):
    '''
    plot grid search with bump points overlayed on ramps, color by F_yield
    Arguments:
        F_slip_w (vector): Vector of F_slip data for ramps, whole square
        F_yield_w (vector): Vector of F_yield data for ramps, whole square
        slopeL_* (vector): Vector of slopeL data for ramps or bumps
        slopeH_* (vector): Vector of slopeH data for ramps or bumps
        F_slip_* (vector): Vector of F_slip data for ramps or bumps
    '''
    # color bar is based on the ranges of F_slip and F_yield for ramps whole square
    vmin = min(np.min(F_slip_w), np.min(F_yield_w))
    vmax = max(np.max(F_slip_w), np.max(F_yield_w))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Base grid plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(slopeL_r, slopeH_r, c=F_yield_r, cmap='viridis', s=60, edgecolor='none', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\mathrm{F}_{\mathrm{yield}}$', fontsize=12)

    # overlay bump points
    plt.scatter(slopeL_b, slopeH_b, c=F_yield_b, cmap='viridis', s=20, norm=norm, marker='o', edgecolor='black', linewidths=0.5)

    # labels
    plt.suptitle(r'$\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes')
    plt.title(r'Overlay bump optimization for $\alpha=1$', fontsize=10)
    plt.xlabel('slopeL (eV/cofactor)')
    plt.ylabel('slopeH (eV/cofactor)')
    plt.tight_layout()
    plt.legend()
    plt.show()

def grid_Fslip_ratio(slopeL, slopeH, F_slip_diff):
    '''
    plot grid search with points colored by the relative change in F_slip (bump/ramp)
    '''
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

def grid_Fyield_ratio(slopeL, slopeH, F_yield_diff):
    '''
    plot grid search with points colored by the relative change in F_slip (bump/ramp)
    '''
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

def pH1_trend(pH1_disp, F_slip_diff):
    '''
    Plot relative improvement in F_slip (bump/ramp) vs. displacement of H1. Plots mean and error bars
    '''
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

def pH1_scatter(pH1_disp, F_slip_diff, slopeL, slopeH):
    '''
    Plot relative improvement in F_slip (bump/ramp) vs. displacement of H1. Plots all points colored by the ratio slopeL/slopeH
    '''
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pH1_disp, F_slip_diff, c=abs(slopeL)/abs(slopeH), s=35)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\frac{|\mathrm{slopeL}|}{|\mathrm{slopeH}|}$', fontsize=12)

    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. H1 Displacement')
    plt.yscale('log')
    plt.xlabel('H1 displacement (bump - ramp) (eV)')
    plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ (bump/ramp)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def constant_bump_v_ramp(slopeL, F_slip_diff):
    '''
    Plot relative change in F_slip with fixed HPB at varying slopeL
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(slopeL, F_slip_diff, color='blue')
    plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    plt.text(x=slopeL.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. Low Potential Branch Slope')
    plt.title('slopeH = -0.15, H1 displacement = 0.075')
    plt.xlabel('Low Potential Branch Slope (eV/cofactor)')
    plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$')
    plt.tight_layout()
    plt.show()

def Nfn1_slopeH_slip(df_bump, df_ramp):
    '''
    Vary slopeH in Nfn-1 and plot to compare F_slip for bump and no bump 
    '''
    common_slopeH = set(df_bump['slopeH']).intersection(set(df_ramp['slopeH']))

    df_bump = df_bump[df_bump["slopeH"].isin(common_slopeH)].reset_index(drop=True)
    df_ramp = df_ramp[df_ramp["slopeH"].isin(common_slopeH)].reset_index(drop=True)

    F_slip_diff = df_bump['F_slip'] / df_ramp['F_slip']

    slopeH = df_bump['slopeH'] # same for both because of intersection

    plt.figure(figsize=(8, 6))
    plt.scatter(slopeH, F_slip_diff, color='blue', s = 5)
    #plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    #plt.text(x=slopeH.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. Nfn-1 High Potential Branch Slope')
    plt.xlabel('High Potential Branch Slope (eV/cofactor)')
    plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$')
    plt.tight_layout()
    plt.show()

def Nfn1_slopeH_yield(df_bump, df_ramp):
    '''
    Vary slopeH in Nfn-1 and plot to compare F_yield for bump and no bump 
    '''
    common_slopeH = set(df_bump['slopeH']).intersection(set(df_ramp['slopeH']))

    df_bump = df_bump[df_bump["slopeH"].isin(common_slopeH)].reset_index(drop=True)
    df_ramp = df_ramp[df_ramp["slopeH"].isin(common_slopeH)].reset_index(drop=True)

    F_yield_diff = df_bump['F_yield'] / df_ramp['F_yield']

    slopeH = df_bump['slopeH'] # same for both because of intersection

    plt.figure(figsize=(8, 6))
    plt.scatter(slopeH, F_yield_diff, color='blue', s = 5)
    #plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    #plt.text(x=slopeL.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$ vs. Low Potential Branch Slope')
    plt.xlabel('High Potential Branch Slope (eV/cofactor)')
    plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$')
    plt.tight_layout()
    plt.show()

def Nfn1_slopeH_Fslip_distrib(df):
    '''
    plot distribution of Fslip
    '''
    F_slip = df['F_slip']

    plt.figure(figsize=(8, 6))
    plt.hist(F_slip, bins=30, color='skyblue', edgecolor='black')
    plt.title(r'Distribution of $\mathrm{F}_{\mathrm{slip}}$')
    plt.xlabel(r'$\mathrm{F}_{\mathrm{slip}}$')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def Nfn1_slopeH_Fyield_distrib(df):
    '''
    plot distribution of Fyield
    '''
    F_yield = df['F_yield']

    plt.figure(figsize=(8, 6))
    plt.hist(F_yield, bins=30, color='skyblue', edgecolor='black')
    plt.title(r'Distribution of $\mathrm{F}_{\mathrm{yield}}$')
    plt.xlabel(r'$\mathrm{F}_{\mathrm{yield}}$')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # === ramps w/ bump overlay ===
    '''
    ramps_w = pd.read_csv("ramps_whole_20250612.csv")
    F_slip_w = ramps_w["F_slip"]
    F_yield_w = ramps_w["F_yield"]

    ramps = pd.read_csv("ramps_whole_20250612.csv")
    slopeL_r = ramps["slopeL"]
    slopeH_r = ramps["slopeH"]
    F_slip_r = ramps["F_slip"]
    F_yield_r = ramps["F_yield"]

    bump = pd.read_csv('BestBump_alpha1_whole_20250618.csv')
    slopeL_b = bump['slopeL']
    slopeH_b = bump['slopeH']
    F_slip_b = bump["F_slip"]
    F_yield_b = bump["F_yield"]
   
    grid_Fslip(F_slip_w, F_yield_w, slopeL_r, slopeH_r, slopeL_b, slopeH_b, F_slip_r, F_slip_b)
    grid_Fyield(F_slip_w, F_yield_w, slopeL_r, slopeH_r, slopeL_b, slopeH_b, F_yield_r, F_yield_b)
    '''
    
    '''
    # === bump vs ramp plots ===
    ramps = pd.read_csv("ramps_corner_bif_20250612.csv")
    bump = pd.read_csv("BestBump_alpha1_corner_bif_20250618.csv")
    ramps_coords = ramps[["slopeL", "slopeH"]].to_numpy()
    bump_coords = bump[["slopeL", "slopeH"]].to_numpy()

    # nearest neighbor matching (since grid searches aren't the same size)
    tree = cKDTree(ramps_coords)
    dists, indices = tree.query(bump_coords)

    pH1_b = bump['potential_H1'].to_numpy()
    pH1_r = ramps['potential_H1'].to_numpy()[indices]
    pH1_disp = pH1_b - pH1_r

    F_slip_b = bump["F_slip"].to_numpy()
    F_slip_r = ramps["F_slip"].to_numpy()[indices]
    F_slip_diff = F_slip_b / F_slip_r

    slopeL = bump['slopeL']
    slopeH = bump['slopeH']

    pH1_scatter(pH1_disp, F_slip_diff, slopeL, slopeH)
    '''

    '''
    # === constant bump vs. ramp ===
    ramps = pd.read_csv("ramps_whole_bif_20250612.csv")
    bump = pd.read_csv("bump_constant_whole_bif_20250613.csv")
    ramps_filtered = ramps[ramps['slopeH'] == -0.15]
    ramps_coords = ramps_filtered['slopeL'].to_numpy().reshape(-1, 1)
    bump_coords = bump['slopeL'].to_numpy().reshape(-1, 1)

    tree = cKDTree(ramps_coords)
    dists, indices = tree.query(bump_coords)

    F_slip_b = bump["F_slip"].to_numpy()
    F_slip_r = ramps["F_slip"].to_numpy()[indices]
    F_slip_diff = F_slip_b / F_slip_r

    slopeL = bump['slopeL']

    constant_bump_v_ramp(slopeL, F_slip_diff)
    '''
    
    # === Nfn-1 vary slopeH ===
    df_bump = pd.read_csv("Nfn1_varyL_bump_bif_20250620.csv")
    #df_bump = df_bump[(df_bump['slopeH'] > -0.159) & (df_bump['slopeH'] < -0.157)]
    df_ramp = pd.read_csv("Nfn1_varyL_ramp_bif_20250620.csv")
    #df_ramp = df_ramp[(df_ramp['slopeH'] > -0.159) & (df_ramp['slopeH'] < -0.157)]
    Nfn1_slopeH_slip(df_bump, df_ramp)
    Nfn1_slopeH_yield(df_bump, df_ramp)
    