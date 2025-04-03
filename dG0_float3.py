'''
BayOpt_DHL_dG.py
Author: Emily Wang
March 2025
How does the optimized landscape at varying levels of high yield vs. low slippage change with dG?
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

def obj_func_full(potentials):
    '''
    Run MEK and return Euclidean distance to optimize
    Arguments:
        potentials {list} -- Contains the passed parameters
    Returns:
        SSR {float} -- (measure of bifurcation efficiency) SSR for distance from 100% efficient bifurcation -1 : 0.5 : 0.5 for flux_D : flux_HR : flux_LR
        events {float} -- number of total bifurcation events 
        flux_ {float} -- fluxes at the reservoirs
    '''
    # fixed potentials
    pL1 = -0.3
    pL2 = -0.2
    pH1 = 0.3
    pH2 = 0.2

    # extract floating potentials
    pL3 = potentials[0][0]
    pD1 = potentials[0][1] # first reduction potential (negative)
    pD2 = -pD1 # second reduction potential is set to negative of first (positive)
    pH3 = -pL3 # dG = -(L3 + H3)

    # run MEK - currently set to open conformation for open flow to low potential branch
    # fixed hyperparameters
    res_rate = 50 # Ortiz JBC 2023
        
    # Time step variables
    N = 20 # time steps
    ztime = 0.00001 # intial time
    dt = 7/(N-1) # controls the orders of magntidue explored (7 in this case)

    net = Network()

    # Cofactor names and potentials
    L1 = Cofactor("L1", [pL1])
    L2 = Cofactor("L2", [pL2])
    L3 = Cofactor("L3", [pL3])
    D = Cofactor("D", [pD1, pD2])
    H1 = Cofactor("H1", [pH1])
    H2 = Cofactor("H2", [pH2])
    H3 = Cofactor("H3", [pH3])

    # Telling the network about the cofactors. Naming cofactors to add to network.
    net.addCofactor(L1)
    net.addCofactor(L2)
    net.addCofactor(L3)
    net.addCofactor(D)
    net.addCofactor(H1)
    net.addCofactor(H2)
    net.addCofactor(H3)

    # Pairwise distances between cofactors. State the distances between cofactor pairs. angstroms
    net.addConnection(D, L1, 10)
    net.addConnection(L1, L2, 10)
    net.addConnection(L2, L3, 10)
    net.addConnection(D, H1, 10)
    net.addConnection(H1, H2, 10)
    net.addConnection(H2, H3, 10)
    net.addConnection(D, L2, 20)  
    net.addConnection(D, H2, 20)
    net.addConnection(D, L3, 30)  
    net.addConnection(D, H3, 30)
    net.addConnection(L1, H1, 20)
    net.addConnection(L1, H2, 30)
    net.addConnection(L1, H3, 40)
    net.addConnection(L2, H1, 30)
    net.addConnection(L2, H2, 40)
    net.addConnection(L2, H3, 50)
    net.addConnection(L3, H1, 40)
    net.addConnection(L3, H2, 50)
    net.addConnection(L3, H3, 60)

    # Infinite reservoirs
    # names, cofactor it is connected to, Redox state of the cofactor, number of electrons transferred, dG between reservoir and cofactor, rate from the cofactor to the reservoir
    net.addReservoir("DR", D, 2, 2, 0.20, res_rate)
    net.addReservoir("LR", L3, 1, 1, 0, res_rate)
    net.addReservoir("HR", H3, 1, 1, 0, res_rate)

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
    events = 10 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

    return SSR, events, fluxD, fluxHR, fluxLR

def obj_func_weighted(potentials, alpha):
    '''
    Tradeoff function to optimize (supports batch inputs)
    Arguments:
        potentials {list (1 searcher) or array (multiple searchers)} -- potentials proposed in an iteration of the optimizer
        alpha {float} -- value in [0,1] denoting the weights of importance for bifurcation efficiency vs. number of bifurcation events. 
                         alpha (efficiency) and (1 - alpha) (events). Default 0.5
    '''
    opt_results = []
    for p in potentials:
        SSR, events, _, _, _, = obj_func_full([p])
        weighted_result = (alpha * SSR) + ((1 - alpha) * (events)) # minimize
        opt_results.append([weighted_result])
    return np.array(opt_results)

def wrapped_obj_func(potentials, alpha):
    '''
    wrapped function for optimizer
    '''
    output = obj_func_weighted(potentials, alpha = alpha)
    #output = obj_func_weighted(potentials, alpha = user_alpha) # use this for user input alpha
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
    potentials = []
    alpha = alpha
    
    # need to change the iteration count based on t test
    for t in range(200):
        seed(t * 100 + 500)
        bounds = [
            {'name': 'L3', 'type': 'continuous', 'domain': (-0.400, -0.05)},
            {'name': 'D1', 'type': 'continuous', 'domain': (-0.500, -0.300)}]

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
        best_potentials = [optimizer.x_opt]
        SSR, events, fluxD, fluxHR, fluxLR = obj_func_full(best_potentials)
        F = (alpha * SSR) + ((1 - alpha) * events)

        # add data to storage vectors
        F_output.append(F) # F for t-th iteration
        F_eff.append(SSR)
        F_amnt.append(events)
        FluxD.append(abs(fluxD))
        FluxHR.append(fluxHR)
        FluxLR.append(fluxLR)
        potentials.append(best_potentials)

    # find best trial and save data
    min_idx, F_t = min(enumerate(F_output), key=lambda x: x[1])
    min_idx = int(min_idx)
    F_eff_best = F_eff[min_idx]
    F_amnt_best = F_amnt[min_idx]
    fluxD_best = FluxD[min_idx]
    fluxHR_best = FluxHR[min_idx]
    fluxLR_best = FluxLR[min_idx]
    best_potentials = potentials[min_idx]
    potH3 = -best_potentials[0][0]

    return [alpha, F_t, F_eff_best, F_amnt_best, fluxD_best, fluxHR_best, fluxLR_best, best_potentials[0][1], best_potentials[0][0], potH3]

if __name__ == '__main__':
    t_start = time.time()
    num_cpus = detect_cpus()
    #print(num_cpus)
    alphas = np.linspace(0, 1, 50)

    with Pool(processes = num_cpus) as pool:
        results = pool.map(run_single_job, alphas)

    # save data
    columns = ["alpha", "F_t", "F_eff_best", "F_amnt_best", "abs(fluxD)", "fluxHR", "fluxLR", "potential_D1", "potential_L3", "potential_H3"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("dG0_20250328.csv", index=False)
    
    t_end = time.time()
    runtime = t_end - t_start
    print("runtime: ", runtime)
