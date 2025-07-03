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
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.cm as cm
import seaborn as sns

def grid_Fsc(ramp, ramp_c, bump_Fsc, bump_Fsc_c, bump_Fyield, bump_Fyield_c):
    '''
    *** need to modify the function based on which data set is being plotted 
    plot grid search with bump points overlayed on ramps, color by F_slip
    Arguments:
        ramp: data frame for ramps
        bump_Fsc: data frame for bump, optimizing for Fsc
        bump_Fyield: data frame for bump, optimizing for Fyield
        ramp_c: data frame for ramps, corner zoom
        bump_Fsc_c: data frame for bump_Fsc, corner zoom
        bump_Fyield_c: data frame for bump_Fyield, corner zoom
    '''

    # color bar 
    vmin = min(np.min(ramp['F_sc']), np.min(bump_Fsc['F_sc']), np.min(bump_Fyield['F_sc']), np.min(ramp_c['F_sc']), np.min(bump_Fsc_c['F_sc']), np.min(bump_Fyield_c['F_sc']))
    vmax = max(np.max(ramp['F_sc']), np.max(bump_Fsc['F_sc']), np.max(bump_Fyield['F_sc']), np.max(ramp_c['F_sc']), np.max(bump_Fsc_c['F_sc']), np.max(bump_Fyield_c['F_sc']))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Base grid plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(ramp_c['slopeL'], ramp_c['slopeH'], c=ramp_c['F_sc'], cmap='viridis', s=60, edgecolor='none', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\mathrm{F}_{\mathrm{sc}}$', fontsize=12)

    # overlay bump points
    plt.scatter(bump_Fyield_c['slopeL'], bump_Fyield_c['slopeH'], c=bump_Fyield_c['F_sc'], cmap='viridis', s=20, marker='o', edgecolor='black', linewidths=0.5, norm=norm)

    # labels
    plt.suptitle(r'$\mathrm{F}_{\mathrm{sc}}$ by ET branch slopes', fontsize=16)
    plt.title(r'Overlay bump optimization for $\mathrm{F}_{\mathrm{yield}}$', fontsize=12)
    plt.xlabel('slopeL (eV/cofactor)', fontsize=16)
    plt.ylabel('slopeH (eV/cofactor)', fontsize=16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.show()

def grid_Fyield(ramp, ramp_c, bump_Fsc, bump_Fsc_c, bump_Fyield, bump_Fyield_c):
    '''
    plot grid search with bump points overlayed on ramps, color by F_yield
    Arguments:
        ramp: data frame for ramps
        bump_Fsc: data frame for bump, optimizing for Fsc
        bump_Fyield: data frame for bump, optimizing for Fyield
        ramp_c: data frame for ramps, corner zoom
        bump_Fsc_c: data frame for bump_Fsc, corner zoom
        bump_Fyield_c: data frame for bump_Fyield, corner zoom
    '''
    # take positive of F_yield values to make log scale work
    ramp_pos = -ramp['F_yield']
    bump_Fsc_pos = -bump_Fsc['F_yield']
    bump_Fyield_pos = -bump_Fyield['F_yield']
    ramp_c_pos = -ramp_c['F_yield']
    bump_Fsc_c_pos = -bump_Fsc_c['F_yield']
    bump_Fyield_c_pos = -bump_Fyield_c['F_yield']

    #print(np.min(ramp_pos), np.min(bump_Fsc_pos), np.min(bump_Fyield_pos), np.min(ramp_c_pos), np.min(bump_Fsc_c_pos), np.min(bump_Fyield_c_pos))
    #print(np.max(ramp_pos), np.max(bump_Fsc_pos), np.max(bump_Fyield_pos), np.max(ramp_c_pos), np.max(bump_Fsc_c_pos), np.max(bump_Fyield_c_pos))
    
    # get min and max of color bar for consistent color bar across all data sets
    #vmin = min(np.min(ramp_pos), np.min(bump_Fsc_pos), np.min(bump_Fyield_pos), np.min(ramp_c_pos), np.min(bump_Fsc_c_pos), np.min(bump_Fyield_c_pos))
    vmin = 1e-8
    vmax = max(np.max(ramp_pos), np.max(bump_Fsc_pos), np.max(bump_Fyield_pos), np.max(ramp_c_pos), np.max(bump_Fsc_c_pos), np.max(bump_Fyield_c_pos))
    norm = LogNorm(vmin=vmin, vmax=vmax) # log scale

    # Base grid plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(ramp['slopeL'], ramp['slopeH'], c=ramp_pos, cmap='viridis', s=60, edgecolor='none', norm=norm)

    # overlay bump points
    plt.scatter(bump_Fyield['slopeL'], bump_Fyield['slopeH'], c=bump_Fyield_pos, cmap='viridis', s=20, marker='o', edgecolor='black', linewidths=0.5, norm=norm)

    cbar = plt.colorbar(sc)

    # choose tick values
    min_exp = int(np.floor(np.log10(vmin)))
    max_exp = int(np.ceil(np.log10(vmax)))
    tick_exponents = np.arange(min_exp, max_exp)
    tick_values = 10.0 ** tick_exponents

    # Set ticks and labels
    cbar.set_ticks(tick_values)
    tick_labels = [fr"$-10^{{{e}}}$" for e in tick_exponents]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(r'$\mathrm{F}_{\mathrm{yield}}$', fontsize=12)

    # labels
    plt.suptitle(r'$\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes', fontsize=16)
    plt.title(r'Overlay bump optimization for $\mathrm{F}_{\mathrm{yield}}$', fontsize=12)
    plt.xlabel('slopeL (eV/cofactor)', fontsize=16)
    plt.ylabel('slopeH (eV/cofactor)', fontsize=16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.show()

def grid_Fsc_ratio(ramp, bump):
    '''
    plot grid search with points colored by the relative change in F_sc (bump/ramp)
    ramp: df for ramp
    bump: df for bump
    '''
    # nearest neighbor search
    ramp_coords = ramp[['slopeL', 'slopeH']].to_numpy()
    bump_coords = bump[['slopeL', 'slopeH']].to_numpy()
    tree = cKDTree(ramp_coords)
    dists, indices = tree.query(bump_coords)

    F_sc_b = bump["F_sc"].to_numpy()
    F_sc_r = ramp["F_sc"].to_numpy()[indices]
    F_sc_ratio = F_sc_b / F_sc_r

    slopeL = bump['slopeL']
    slopeH = bump['slopeH']

    # log color bar
    vmin = 1e-3
    vmax = 1.3
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # make plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(slopeL, slopeH, c=F_sc_ratio, cmap='viridis', s=60, edgecolor='none', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\frac{\mathrm{F}_{\mathrm{sc}} \, \mathrm{bump}}{\mathrm{F}_{\mathrm{sc}} \, \mathrm{ramp}}$', fontsize=12)

    # labels
    plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{sc}}$ by ET branch slopes', fontsize=16)
    plt.title(r'bump optimized for $\mathrm{F}_{\mathrm{sc}}$', fontsize=12)
    plt.xlabel('slopeL (eV/cofactor)', fontsize=14)
    plt.ylabel('slopeH (eV/cofactor)', fontsize=14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.show()

def grid_Fyield_ratio(ramp, bump):
    '''
    plot grid search with points colored by the relative change in F_yield (bump/ramp)
    ramp: df for ramp
    bump: df for bump
    '''
    # nearest neighbor search
    ramp_coords = ramp[['slopeL', 'slopeH']].to_numpy()
    bump_coords = bump[['slopeL', 'slopeH']].to_numpy()
    tree = cKDTree(ramp_coords)
    dists, indices = tree.query(bump_coords)

    F_yield_b = bump["F_yield"].to_numpy()
    F_yield_r = ramp["F_yield"].to_numpy()[indices]
    F_yield_ratio = F_yield_b / F_yield_r

    slopeL = bump['slopeL']
    slopeH = bump['slopeH']

    # log color bar
    vmin = 1e-3
    vmax = 1.3
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # make plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(slopeL, slopeH, c=F_yield_ratio, cmap='viridis', s=60, edgecolor='none', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\frac{\mathrm{F}_{\mathrm{yield}} \, \mathrm{bump}}{\mathrm{F}_{\mathrm{yield}} \, \mathrm{ramp}}$', fontsize=12)

    # labels
    plt.suptitle(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$ by ET branch slopes', fontsize=16)
    plt.title(r'bump optimized for $\mathrm{F}_{\mathrm{sc}}$', fontsize=12)
    plt.xlabel('slopeL (eV/cofactor)', fontsize=14)
    plt.ylabel('slopeH (eV/cofactor)', fontsize=14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.legend(fontsize=12)
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

def pH1_scatter(pH1_disp, F_sc_diff, slopeL, slopeH):
    '''
    Plot relative improvement in F_sc (bump/ramp) vs. displacement of H1. Plots all points colored by the ratio slopeL/slopeH
    '''
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pH1_disp, F_sc_diff, c=slopeL/slopeH, s=35)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\frac{\mathrm{slopeL}}{\mathrm{slopeH}}$', fontsize=12)

    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{sc}}$ vs. H1 Displacement')
    plt.yscale('log')
    plt.xlabel('H1 displacement (bump - ramp) (eV)')
    plt.ylabel(r'Relative $\Delta\mathrm{F}_{\mathrm{sc}}$ (bump/ramp)')
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
    plt.scatter(slopeH, F_slip_diff, color='blue', s = 10)
    #plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    #plt.text(x=slopeH.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{slip}}$ vs. High Potential Branch Slope', fontsize=16)
    plt.xlabel('High Potential Branch Slope (eV/cofactor)', fontsize=14)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{slip}} \, \mathrm{bump}}{\mathrm{F}_{\mathrm{slip}} \, \mathrm{ramp}}$', fontsize=14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
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
    plt.scatter(slopeH, F_yield_diff, color='blue', s = 10)
    #plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    #plt.text(x=slopeL.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{yield}}$ vs. High Potential Branch Slope', fontsize=16)
    plt.xlabel('High Potential Branch Slope (eV/cofactor)', fontsize=14)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{yield}} \, \mathrm{bump}}{\mathrm{F}_{\mathrm{yield}} \, \mathrm{ramp}}$', fontsize=14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.tight_layout()
    plt.show()

def Nfn1_slopeH_sc(df_bump, df_ramp):
    '''
    Vary slopeH in Nfn-1 and plot to compare F_sc for bump and no bump 
    '''
    common_slopeH = set(df_bump['slopeH']).intersection(set(df_ramp['slopeH']))

    df_bump = df_bump[df_bump["slopeH"].isin(common_slopeH)].reset_index(drop=True)
    df_ramp = df_ramp[df_ramp["slopeH"].isin(common_slopeH)].reset_index(drop=True)

    F_yield_diff = df_bump['F_sc'] / df_ramp['F_sc']

    slopeH = df_bump['slopeH'] # same for both because of intersection

    plt.figure(figsize=(8, 6))
    plt.scatter(slopeH, F_yield_diff, color='blue', s = 10)
    #plt.axhline(1, linestyle='--', color='red')  # Reference line: no improvement
    #plt.text(x=slopeL.max(), y=1 - 0.02, s='no improvement', ha='right', va='top', fontsize=10, color='red')
    plt.title(r'Relative $\Delta\mathrm{F}_{\mathrm{sc}}$ vs. High Potential Branch Slope', fontsize=16)
    plt.xlabel('High Potential Branch Slope (eV/cofactor)', fontsize=14)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{sc}} \, \mathrm{bump}}{\mathrm{F}_{\mathrm{sc}} \, \mathrm{ramp}}$', fontsize=14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
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

def Nfn1_bump_Fslip(df):
    '''
    Plot ratio Fslip (bump) / F_slip (ramp) for a range of bump sizes
    '''
    pH1 = df['pH1']
    F_slip_b = df['F_slip']
    F_slip_r = 0.03346239204029306

    F_slip_ratio = F_slip_b / F_slip_r

    plt.figure(figsize=(8, 6))
    plt.plot(pH1, F_slip_ratio, color='blue', lw = 4)
    plt.title(r'$\mathrm{F}_{\mathrm{slip}}$ ratio vs. potential on H1', fontsize=22)
    plt.xlabel('Potential on H1 (eV)', fontsize=22)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{slip}} \mathrm{bump}}{\mathrm{F}_{\mathrm{slip}} \mathrm{ramp}}$', fontsize=22)
    plt.yscale('log')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.show()

def Nfn1_bump_Fyield(df):
    '''
    Plot ratio Fyield (bump) / F_yield (ramp) for a range of bump sizes
    '''
    pH1 = df['pH1']
    F_yield_b = df['F_yield']
    F_yield_r = - 1 / 46.74095979300153

    F_yield_ratio = F_yield_b / F_yield_r

    plt.figure(figsize=(8, 6))
    plt.plot(pH1, F_yield_ratio, color='blue', lw = 4)
    plt.title(r'$\mathrm{F}_{\mathrm{yield}}$ ratio vs. potential on H1', fontsize=22)
    plt.xlabel('Potential on H1 (eV)', fontsize=22)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{yield}} \mathrm{bump}}{\mathrm{F}_{\mathrm{yield}} \mathrm{ramp}}$', fontsize=22)
    plt.yscale('log')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.show()

def Nfn1_bump_Fsc(df):
    '''
    Plot ratio Fslip (bump) / F_slip (ramp) for a range of bump sizes
    '''
    pH1 = df['pH1']
    F_sc_b = df['F_sc']
    F_sc_r = 1.1945713514691109e-05

    F_sc_ratio = F_sc_b / F_sc_r

    plt.figure(figsize=(8, 6))
    plt.plot(pH1, F_sc_ratio, color='blue', lw = 4)
    plt.title(r'$\mathrm{F}_{\mathrm{sc}}$ ratio vs. potential on H1', fontsize=22)
    plt.xlabel('Potential on H1 (eV)', fontsize=22)
    plt.ylabel(r'$\frac{\mathrm{F}_{\mathrm{sc}} \mathrm{bump}}{\mathrm{F}_{\mathrm{sc}} \mathrm{ramp}}$', fontsize=22)
    plt.yscale('log')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    '''
    # === ramps w/ bump overlay ===
    ramp = pd.read_csv("ramps_FscFyield_20250630.csv")
    ramp_c = pd.read_csv('ramps_FscFyield_corner_20250702.csv')
    bump_Fsc = pd.read_csv('bump_Fsc_20250630.csv')
    bump_Fsc_c = pd.read_csv('bump_Fsc_corner_20250701.csv')
    bump_Fyield = pd.read_csv('bump_Fyield_20250629.csv')
    bump_Fyield_c = pd.read_csv('bump_Fyield_corner_20250701.csv')
   
    # overlay plots
    #grid_Fsc(ramp, ramp_c, bump_Fsc, bump_Fsc_c, bump_Fyield, bump_Fyield_c)
    #grid_Fyield(ramp, ramp_c, bump_Fsc, bump_Fsc_c, bump_Fyield, bump_Fyield_c)

    # ratio plots 
    # for bump opt by Fsc vmin = 1e-3, vmax = 1.3
    # for bump opt by Fyield vmin = 0.2, vmax = 22.2
    grid_Fsc_ratio(ramp, bump_Fsc)
    grid_Fyield_ratio(ramp, bump_Fsc)
    grid_Fsc_ratio(ramp_c, bump_Fsc_c)
    grid_Fyield_ratio(ramp_c, bump_Fsc_c)
    '''
    
    '''
    # === bump vs ramp plots ===
    ramps = pd.read_csv("ramps_FscFyield_bif_20250630.csv")
    bump = pd.read_csv("bump_Fsc_bif_20250630.csv")
    ramps_coords = ramps[["slopeL", "slopeH"]].to_numpy()
    bump_coords = bump[["slopeL", "slopeH"]].to_numpy()

    # nearest neighbor matching (since grid searches aren't the same size)
    tree = cKDTree(ramps_coords)
    dists, indices = tree.query(bump_coords)

    pH1_b = bump['potential_H1'].to_numpy()
    pH1_r = ramps['potential_H1'].to_numpy()[indices]
    pH1_disp = pH1_b - pH1_r

    F_sc_b = bump["F_sc"].to_numpy()
    F_sc_r = ramps["F_sc"].to_numpy()[indices]
    F_sc_diff = F_sc_b / F_sc_r

    slopeL = bump['slopeL']
    slopeH = bump['slopeH']

    pH1_scatter(pH1_disp, F_sc_diff, slopeL, slopeH)
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
    df_bump = pd.read_csv("Nfn1_varyH_bump_bif_20250701.csv")
    df_bump = df_bump[(df_bump['slopeH'] > -0.163) & (df_bump['slopeH'] < -0.153)]
    df_ramp = pd.read_csv("Nfn1_varyH_ramp_bif_20250701.csv")
    df_ramp = df_ramp[(df_ramp['slopeH'] > -0.163) & (df_ramp['slopeH'] < -0.153)]
    Nfn1_slopeH_slip(df_bump, df_ramp)
    Nfn1_slopeH_yield(df_bump, df_ramp)
    Nfn1_slopeH_sc(df_bump, df_ramp)
    
    '''
    # === Nfn-1 metrics for range of bump sizes vs. ramp ====
    df = pd.read_csv('Nfn1_varybump_metrics_20250630.csv')
    Nfn1_bump_Fslip(df)
    Nfn1_bump_Fyield(df)
    Nfn1_bump_Fsc(df)
    '''