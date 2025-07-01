'''
bump_Fsc
Author: Emily Wang
June 29, 2025
Uses toy simplified Nfn1 model. D midpoint potential is 0 meV, LPB is linear, and all distances are 10, 
but reservoir placement and D gap is like Nfn1. Updating F_slip --> F_sc, defined by the short circuit fluxes
'''

# Import Modules
# modules for MEK
from MEK_public import *
import math

import numpy as np
from numpy.linalg import eig
from numpy.random import seed

# optimizer
import GPyOpt
from GPyOpt.methods import BayesianOptimization

# standard packages
import matplotlib.pyplot as plt
import pandas as pd
import time

import os

def obj_func_full(potentials, slopes):
    '''
    full objective function, takes BayOpt parameter(s) and effective slopes on LPB and HPB
    '''
    # floating potential (bump)
    pH1 = potentials[0][0]

    # extract floating potentials
    pD1 = -0.4755 # first reduction potential (negative)
    pD2 = 0.4755 # second reduction potential is set to negative of first (positive)
    pL1 = pD1 + slopes[0]
    pL2 = pD1 + 2 * slopes[0]
    pH2 = pD2 + 2 * slopes[1]

    net = Network()

    # Cofactor names and potentials
    L1 = Cofactor("L1", [pL1])
    L2 = Cofactor("L2", [pL2])
    D = Cofactor("D", [pD1, pD2])
    H1 = Cofactor("H1", [pH1])
    H2 = Cofactor("H2", [pH2])

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
    res_rate = 50
    dG_DR = -0.072
    dG_LR = -0.109
    dG_HR = 0.008
    net.addReservoir("DR", D, 2, 2, dG_DR, res_rate)
    net.addReservoir("LR", L2, 1, 1, dG_LR, res_rate)
    net.addReservoir("HR", H2, 1, 1, dG_HR, res_rate)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1
    t = 10 ** 4 # longer time to reach SS
    pop_MEK = net.evolve(t, pop_MEK_init)

    D_flux = net.getReservoirFlux("DR", pop_MEK)
    H_flux = net.getReservoirFlux("HR", pop_MEK)
    L_flux = net.getReservoirFlux("LR", pop_MEK)

    # short circuit pathways
    D1_to_H1 = net.getCofactorFlux(D, 1, H1, 1, pop_MEK)
    L1_to_D2 = net.getCofactorFlux(L1, 1, D, 2, pop_MEK)

    # compute bifurcation metrics
    F_sc = D1_to_H1 + L1_to_D2 # inverse of sum of short circuit fluxes
    F_yield = 1 / (abs(D_flux) + abs(H_flux) + abs(L_flux))

    F = F_sc

    return F, F_sc, F_yield, D_flux, H_flux, L_flux, D1_to_H1, L1_to_D2

def make_wrapped_obj(slopes):
    '''
    wrapper function for obj_func_full
    '''
    def wrapped_obj(potentials):
        results = []
        for i in range(potentials.shape[0]):
            pH1_array = [[potentials[i, 0]]]  # shape (1,1) to match original func
            F, *_ = obj_func_full(pH1_array, slopes)
            results.append([F])  # must be a list of lists
        return np.array(results)
    return wrapped_obj

def run_single_job(slopes):
    '''
    Runs t serial runs of the Bayesian Optimizer and returns the best optimization results

    Args:
    slopes (array-like): [slopeL, slopeH]
    '''
    # store data for plotting
    F_output = [] # output F for each of the t trials
    F_sc = []
    F_yield = []
    FluxD = []
    FluxHR = []
    FluxLR = []
    potentials = []
    sc1 = []
    sc2 = []
    
    # need to change the iteration count based on t test
    for t in range(300):
        seed(t * 100 + 500)
        bounds = [{'name': 'H1', 'type': 'continuous', 'domain': (0.275, 0.675)}]

        maxiter = 30
        wrapped = make_wrapped_obj(slopes)

        # Initialize the optimizer
        optimizer = BayesianOptimization(
            f = wrapped, 
            domain = bounds,
            acquisition_type = 'EI',
            initial_design_num = 20,
            initial_design_type = 'random'
        )

        # modify acquisition function to include exploration parameter (jitter)
        optimizer.acquisition = GPyOpt.acquisitions.AcquisitionMPI(
            model = optimizer.model,
            space = optimizer.space,
            optimizer = optimizer.acquisition_optimizer,
            jitter = 5  # Higher jitter increases exploration
            )

        # Perform Bayesian Optimization
        optimizer.run_optimization(max_iter = maxiter)

        # Evaluate the fluxes using the optimized parameters
        best_potentials = [optimizer.x_opt]
        F_val, F_sc_val, F_yield_val, fluxD, fluxHR, fluxLR, D_H1_flux, L1_D_flux = obj_func_full(best_potentials, slopes)

        # add data to storage vectors
        F_output.append(F_val) # F for t-th iteration
        F_sc.append(F_sc_val)
        F_yield.append(F_yield_val)
        FluxD.append(fluxD)
        FluxHR.append(fluxHR)
        FluxLR.append(fluxLR)
        potentials.append(best_potentials)
        sc1.append(D_H1_flux)
        sc2.append(L1_D_flux)

    # find best trial and save data
    min_idx, F_t = min(enumerate(F_output), key=lambda x: x[1])
    min_idx = int(min_idx)
    F_sc_best = F_sc[min_idx]
    F_yield_best = F_yield[min_idx]
    fluxD_best = FluxD[min_idx]
    fluxHR_best = FluxHR[min_idx]
    fluxLR_best = FluxLR[min_idx]
    sc1_best = sc1[min_idx]
    sc2_best = sc2[min_idx]
    best_potentials = potentials[min_idx]

    return [slopes[0], slopes[1], F_t, F_sc_best, F_yield_best, fluxD_best, fluxHR_best, fluxLR_best, best_potentials[0][0], sc1_best, sc2_best]

if __name__ == '__main__':
    t_start = time.time()
    timestr = time.strftime("%Y%m%d")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    grid_size = 25  # 25x25 grid for 625 points
    slopeL_vals = np.linspace(-0.200, 0.200, grid_size)
    slopeH_vals = np.linspace(-0.200, 0.200, grid_size)
    slopeL_grid, slopeH_grid = np.meshgrid(slopeL_vals, slopeH_vals)
    slope_pairs = np.column_stack([slopeL_grid.ravel(), slopeH_grid.ravel()])
    chunk = np.array_split(slope_pairs, num_tasks)[task_id]

    results = [run_single_job(slopes) for slopes in chunk]

    # save data
    columns = ["slopeL", "slopeH", "F", "F_sc", "F_yield", "fluxD", "fluxHR", "fluxLR", "potential_H1", "D_H1_flux", "L1_D_flux"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"bump_Fsc_{task_id}_"+timestr+".csv", index=False)
    
    t_end = time.time()
    runtime = t_end - t_start
    print(f"Total runtime: {runtime:.2f} seconds")
    

