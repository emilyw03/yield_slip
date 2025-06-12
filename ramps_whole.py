'''
2cof_MEK.py
Author: Emily Wang
April 2025

Basic MEK script for manual testing.
'''

# Import Modules
# modules for MEK
from Fncts_newresrate import *
import math

import numpy as np
from numpy.linalg import eig

# standard packages
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

def obj_func_full(slopeL, slopeH):
    '''
    Run MEK and return Euclidean distance to optimize
    Arguments:
        slope {list} -- [slopeL, slopeH]
    Returns:
        SSR {float} -- (measure of bifurcation efficiency) SSR for distance from 100% efficient bifurcation -1 : 0.5 : 0.5 for flux_D : flux_HR : flux_LR
        events {float} -- number of total bifurcation events 
        flux_ {float} -- fluxes at the reservoirs
    '''
    # run MEK
    # fixed hyperparameters
    res_rate = 50 # rate for cofactor --> reservoir ET; adjusted for V_0 = 0.01
        
    # Time step variables
    N = 20 # time steps
    ztime = 0.00001 # intial time
    dt = 7/(N-1) # controls the orders of magntidue explored (7 in this case)

    net = Network()

    # Cofactor names and potentials
    D = Cofactor("D", [-0.3, 0.3])   #inverted
    L1 = Cofactor("L1", [-0.3 + slopeL*1])
    L2 = Cofactor("L2", [-0.3 + slopeL*2])
    H1 = Cofactor("H1", [0.3 + slopeH*1])
    H2 = Cofactor("H2", [0.3 + slopeH*2])

    # Telling the network about the cofactors. Naming cofactors to add to network.
    net.addCofactor(L1)
    net.addCofactor(L2)
    net.addCofactor(D)
    net.addCofactor(H1)
    net.addCofactor(H2)

    # Pairwise distances between cofactors. State the distances between cofactor pairs. angstroms
    net.addConnection(D, L1, 10)
    net.addConnection(L1, L2, 10)
    net.addConnection(D, H1, 10)
    net.addConnection(H1, H2, 10)
    net.addConnection(D, L2, 20)  
    net.addConnection(D, H2, 20)
    net.addConnection(L1, H1, 20)
    net.addConnection(L1, H2, 30)
    net.addConnection(L2, H1, 30)
    net.addConnection(L2, H2, 40)


    # Infinite reservoirs
    # names, cofactor it is connected to, Redox state of the cofactor, number of electrons transferred, dG between reservoir and cofactor, rate from the cofactor to the reservoir
    net.addReservoir("DR", D, 2, 2, 0.20, res_rate)
    net.addReservoir("LR", L2, 1, 1, 0, res_rate)
    net.addReservoir("HR", H2, 1, 1, 0, res_rate)

    # Build matrix describing all connections and the rate matrix
    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # Set initial conditions (P_0(t))
    # currently set up so there are no electrons in system
    pop_MEK_init = np.zeros(net.num_state)
    pop_MEK_init[0] = 1

    # KINETICS
    # compute fluxes at SS
    # get time value; log scale
    time = ztime*(10**(N*dt))

    # get P(t)
    pop_MEK = net.evolve(time, pop_MEK_init)
    # compute fluxes
    fluxD = net.getReservoirFlux("DR", pop_MEK)
    fluxHR = net.getReservoirFlux("HR", pop_MEK)
    fluxLR = net.getReservoirFlux("LR", pop_MEK)

    # normalize fluxes by largest flux
    fluxes = np.array([fluxD, fluxHR, fluxLR])
    max_flux = np.max(np.absolute(fluxes))
    fluxD_norm = fluxD / abs(max_flux)
    fluxHR_norm = fluxHR / abs(max_flux)
    fluxLR_norm = fluxLR / abs(max_flux)

    # metric for efficiency: SSR from 100% efficient bifurcation flux ratio -1:0.5:0.5
    F_slip = math.sqrt((fluxD_norm + 1) ** 2 + (fluxHR_norm - 0.5) ** 2 + (fluxLR_norm - 0.5) ** 2)

    # metric for bifurcation amount: 1000 / (# of events)
    # numerator selected so that events is of similar magnitude to SSR
    F_yield = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

    return [slopeL, slopeH, F_slip, F_yield, fluxD, fluxHR, fluxLR]

if __name__ == '__main__':
    t_start = time.time()
    timestr = time.strftime("%Y%m%d")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    grid_size = 100  # 100x100 grid for 10,000 points
    slopeL_vals = np.linspace(-0.200, 0.200, grid_size)
    slopeH_vals = np.linspace(-0.200, 0.200, grid_size)
    slopeL_grid, slopeH_grid = np.meshgrid(slopeL_vals, slopeH_vals)
    slope_pairs = np.column_stack([slopeL_grid.ravel(), slopeH_grid.ravel()])
    chunk = np.array_split(slope_pairs, num_tasks)[task_id]

    results_all = []
    for slopeL, slopeH in chunk:
        result = obj_func_full(slopeL, slopeH)

        # adding two extra columns for pH1 and dG
        potential_H1 = 0.300 + slopeH
        pLR = -0.3 + 2*slopeL
        pHR = 0.3 + 2*slopeH
        dG1 = -(pLR + pHR)
        dG2 = -(2*pHR)
        eff = result[6] / result[5] if fluxHR != 0 else 0
        dG = (eff * dG1) + ((1 - eff) * dG2)

        result.extend([potential_H1, dG])
        results_all.append(result)

    # save data
    columns = ["slopeL", "slopeH", "F_slip", "F_yield", "fluxD", "fluxHR", "fluxLR", "potential_H1", "dG"]
    df = pd.DataFrame(results_all, columns=columns)
    df.to_csv(f"ramps_whole_{task_id}_"+timestr+".csv", index=False)
    
    t_end = time.time()
    runtime = t_end - t_start
    print(f"Total runtime: {runtime:.2f} seconds")