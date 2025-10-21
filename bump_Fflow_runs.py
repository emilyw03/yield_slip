'''
bump_Fsc
Author: Emily Wang
October 21, 2025
Test convergence of optimizer
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
    F_sc = D1_to_H1 + L1_to_D2 # sum of short circuit fluxes
    F_yield = -(abs(D_flux) + abs(H_flux) + abs(L_flux))

    F = F_flow

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

def one_run(slopes, t):
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

    return F_flow_val

if __name__ == '__main__':
    t_start = time.time()
    timestr = time.strftime("%Y%m%d")

    slopes = np.array([0.150, -0.150])
    F_flow_data = []
    F_flow_best_data = []
    t = []

    for t in range(300):
        F_flow = one_run(slopes, t)
        F_flow_data.append(F_flow)
        t_data.append(t)
        F_flow_best_data.append(min(F_flow_data))
    
    results = [t_data, F_flow_best_data, F_flow_data]

    columns = ["t", "F_flow_best", "F_flow"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"bump_optFflow_converg_"+timestr+".csv", index=False)

    t_end = time.time()
    runtime = t_end - t_start
    print(f"Total runtime: {runtime:.2f} seconds")
    

