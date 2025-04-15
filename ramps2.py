'''
ramps.py
Author: Emily Wang
March 2025
Enforcing ramp landscape. How does the ramp slope of the optimized landscape change at varying levels of high yield vs. low slippage?
Three cofactors in each of the high and low potential branches.
Constrain L3 and H3 to not reach 0

Bayesian Optimization on simple toy bifurcator DHL from Kiriko's finite reservoir manuscript/Jon Yuly's paper
main script for running the optimizer and plotting a graph of fluxes vs. alpha (relative weight of efficiency factor)
For a given alpha, t runs are serial
Alpha loop is parallelized
Optimizer specs: 
    - uses Gaussian process Bayesian Optimization
    - For each optimization, choose 20 random initial points to construct the likelihood function. Run 30 iterations of optimizer.
    - For one full optimization process, run t = 20 separate optimizations. 
      The final result is the taken from the run that yields the minimum F of 20 runs.
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

# for parallelizing
from multiprocessing import Pool, cpu_count

# standard packages
import matplotlib.pyplot as plt
import pandas as pd
import time

import os

def obj_func_full(slope):
    '''
    Run MEK and return Euclidean distance to optimize
    Arguments:
        slope {list} -- Slope passed from Bayesian Opt
    Returns:
        SSR {float} -- (measure of bifurcation efficiency) SSR for distance from 100% efficient bifurcation -1 : 0.5 : 0.5 for flux_D : flux_HR : flux_LR
        events {float} -- number of total bifurcation events 
        flux_ {float} -- fluxes at the reservoirs
    '''
    # extract params
    slopeL = slope[0][0]
    slopeH = slope[0][1]

    # run MEK
    # fixed hyperparameters
    res_rate = 50 # Ortiz JBC 2023
        
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
    SSR = (fluxD_norm + 1) ** 2 + (fluxHR_norm - 0.5) ** 2 + (fluxLR_norm - 0.5) ** 2

    # metric for bifurcation amount: 1000 / (# of events)
    # numerator selected so that events is of similar magnitude to SSR
    events = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

    return SSR, events, fluxD, fluxHR, fluxLR

def obj_func_weighted(slopes, alpha):
    '''
    Tradeoff function to optimize (supports batch inputs)
    Arguments:
        slope -- slope proposed in an iteration of the optimizer
        alpha {float} -- value in [0,1] denoting the weights of importance for bifurcation efficiency vs. number of bifurcation events. 
                         alpha (efficiency) and (1 - alpha) (events). Default 0.5
    '''
    opt_results = []
    for s in slopes:
        SSR, events, _, _, _, = obj_func_full([s])
        weighted_result = (alpha * SSR) + ((1 - alpha) * (events)) # minimize
        opt_results.append([weighted_result])
    return np.array(opt_results)

def wrapped_obj_func(slopes, alpha):
    '''
    wrapped function for optimizer
    '''
    output = obj_func_weighted(slopes, alpha = alpha)
    return output

def detect_cpus():
    '''
    detect number of CPU cores
    if SLURM_NTASKS set, use that value, else default to system's CPU count
    '''
    return int(os.environ.get("SLURM_NTASKS", cpu_count()))

def run_single_job(alpha):
    '''
    scripts for t serial runs
    '''
    # store data for plotting
    F_output = [] # output F for each of the t trials
    F_eff = []
    F_amnt = []
    FluxD = []
    FluxHR = []
    FluxLR = []
    SLOPE = []
    alpha = alpha
    
    # need to change the iteration count based on t test
    for t in range(200):
        seed(t * 100 + 500)
        bounds = [
            {'name': 'slopeL', 'type': 'continuous', 'domain': (-0.099, 0.099)},
            {'name': 'slopeH', 'type': 'continuous', 'domain': (-0.099, 0.099)}]

        maxiter = 30

        # Initialize the optimizer
        optimizer = BayesianOptimization(
            f = lambda x: wrapped_obj_func(x, alpha), 
            domain = bounds,
            acquisition_type = 'EI',
            initial_design_num = 20,
            initial_design_type = 'random'
        )

        # modify acquisition function to include exploratin parameter (jitter)
        optimizer.acquisition = GPyOpt.acquisitions.AcquisitionMPI(
            model = optimizer.model,
            space = optimizer.space,
            optimizer = optimizer.acquisition_optimizer,
            jitter = 5  # Higher jitter increases exploration
            )

        # Perform Bayesian Optimization
        optimizer.run_optimization(max_iter = maxiter)

        # Evaluate the fluxes using the optimized parameters
        best_slopes = [optimizer.x_opt]
        SSR, events, fluxD, fluxHR, fluxLR = obj_func_full(best_slopes)
        F = (alpha * SSR) + ((1 - alpha) * events)

        # add data to storage vectors
        F_output.append(F) # F for t-th iteration
        F_eff.append(SSR)
        F_amnt.append(events)
        FluxD.append(abs(fluxD))
        FluxHR.append(fluxHR)
        FluxLR.append(fluxLR)
        SLOPE.append(best_slopes)

    # find best trial and save data
    min_idx, F_t = min(enumerate(F_output), key=lambda x: x[1])
    min_idx = int(min_idx)
    F_eff_best = F_eff[min_idx]
    F_amnt_best = F_amnt[min_idx]
    fluxD_best = FluxD[min_idx]
    fluxHR_best = FluxHR[min_idx]
    fluxLR_best = FluxLR[min_idx]
    best_slope = SLOPE[min_idx]

    return [alpha, F_t, F_eff_best, F_amnt_best, fluxD_best, fluxHR_best, fluxLR_best, best_slope[0][0], best_slope[0][1]]

if __name__ == '__main__':
    t_start = time.time()
    num_cpus = detect_cpus()
    #print(num_cpus)
    alphas = np.linspace(0, 1, 50)

    with Pool(processes = num_cpus) as pool:
        results = pool.map(run_single_job, alphas)

    # save data
    columns = ["alpha", "F_t", "F_eff_best", "F_amnt_best", "abs(fluxD)", "fluxHR", "fluxLR", "slopeL", "slopeH"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("ramps2_20250410.csv", index=False)

    t_end = time.time()
    runtime = t_end - t_start
    print("runtime: ", runtime)