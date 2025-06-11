'''
bumpOpt_Bay
Author: Emily Wang
June 5, 2025
Optimize bump size (H1) for a grid of effective slope values. Uses Bayesian Optimization.
Selected alpha = 0.03 for objective function.
'''

# Import Modules
# modules for MEK
from Fncts_newresrate import *
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
    pD1 = -0.300 # first reduction potential (negative)
    pD2 = -pD1 # second reduction potential is set to negative of first (positive)
    pL1 = -0.25
    pL2 = pD1 + 2 * slopes[0]
    pH2 = pD2 + 2 * slopes[1]

    # run MEK - currently set to open conformation for open flow to low potential branch
    # fixed hyperparameters
    res_rate = 50 # Ortiz JBC 2023
        
    # Time step variables
    N = 20 # time steps
    ztime = 0.00001 # initial time
    dt = 7/(N-1) # controls the orders of magnitude explored (7 in this case)

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

    alpha = 1
    F = alpha * F_slip + (1-alpha) * F_yield

    return F, F_slip, F_yield, fluxD, fluxHR, fluxLR

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

def detect_cpus():
    '''
    detect number of CPU cores
    if SLURM_NTASKS set, use that value, else default to system's CPU count
    '''
    return int(os.environ.get("SLURM_NTASKS", cpu_count()))

def run_single_job(slopes):
    '''
    Runs t serial runs of the Bayesian Optimizer and returns the best optimization results

    Args:
    slopes (array-like): [slopeL, slopeH]
    '''
    # store data for plotting
    F_output = [] # output F for each of the t trials
    F_slip = []
    F_yield = []
    FluxD = []
    FluxHR = []
    FluxLR = []
    potentials = []
    
    # need to change the iteration count based on t test
    for t in range(300):
        seed(t * 100 + 500)
        bounds = [{'name': 'H1', 'type': 'continuous', 'domain': (0.05, 0.500)}]

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
        F_val, F_slip_val, F_yield_val, fluxD, fluxHR, fluxLR = obj_func_full(best_potentials, slopes)

        # add data to storage vectors
        F_output.append(F_val) # F for t-th iteration
        F_slip.append(F_slip_val)
        F_yield.append(F_yield_val)
        FluxD.append(fluxD)
        FluxHR.append(fluxHR)
        FluxLR.append(fluxLR)
        potentials.append(best_potentials)

    # find best trial and save data
    min_idx, F_t = min(enumerate(F_output), key=lambda x: x[1])
    min_idx = int(min_idx)
    F_slip_best = F_slip[min_idx]
    F_yield_best = F_yield[min_idx]
    fluxD_best = FluxD[min_idx]
    fluxHR_best = FluxHR[min_idx]
    fluxLR_best = FluxLR[min_idx]
    best_potentials = potentials[min_idx]

    return [slopes[0], slopes[1], F_t, F_slip_best, F_yield_best, fluxD_best, fluxHR_best, fluxLR_best, best_potentials[0][0]]

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
    columns = ["slopeL", "slopeH", "F_t", "F_slip", "F_yield", "fluxD", "fluxHR", "fluxLR", "potential_H1"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"BestBump_alpha1_whole_{task_id}_"+timestr+".csv", index=False)
    
    t_end = time.time()
    runtime = t_end - t_start
    print(f"Total runtime: {runtime:.2f} seconds")
    

