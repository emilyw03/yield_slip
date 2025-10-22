# sandbox.py
# for testing

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

def grid_Fsc(ramp):
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
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    # color bar 
    vmin = np.min(ramp['F_sc'])
    vmax = np.max(ramp['F_sc'])
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Base grid plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(ramp['slopeL'], ramp['slopeH'], c=ramp['F_sc'], cmap='viridis', s=60, edgecolor='none', norm=norm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\mathit{F}_{\mathit{sc}}$', fontsize=12)

    # labels
    plt.title(r'$\mathrm{F}_{\mathrm{sc}}$ by ET branch slopes', fontsize=16)
    plt.xlabel('slopeL (V/cofactor)', fontsize=16)
    plt.ylabel('slopeH (V/cofactor)', fontsize=16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ramp = pd.read_csv("ramps_FscFyield_20250630.csv")
    grid_Fsc(ramp)
