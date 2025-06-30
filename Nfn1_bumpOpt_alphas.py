'''
Nfn1_bumpOpt_alphas.py
Author: Emily Wang
June 29, 2025
Optimize potential on H1 in Nfn1 landscape. Sweep alpha = [0, 1]. Uses new metric, F_sc, based on short circuit fluxes.
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

def obj_func_full(potentials, alpha):
    '''
    full objective function, takes potential on H1 and alpha
    '''
    # floating potential (bump)
    mu_FeS_H1 = potentials[0][0]

    HP_length = 13.1
    LP_length = 5.4
    # mu_NADPH = -0.320 
    mu_NADPH = -0.400
    mu_Fd = -0.420
    mu_NAD = -0.280

    # NADPH_L_FAD_reservoir_rate = 113     # NADPH -> L-FAD rate (reservoir -> cofactor) pH = 7.5
    NADPH_L_FAD_reservoir_rate = 36     # NADPH -> L-FAD rate (reservoir -> cofactor) pH = 9.5
    L2_Fdox_reservoir_rate = 100      # [4Fe-4S] -> Fdox rate  (cofactor -> reservoir)
    S_FAD_NAD_reservoir_rate = 50    # S-FAD -> NAD rate (cofactor -> reservoir)

    net = Network()

    L_FAD = Cofactor("L_FAD", [-0.911, 0.04]) 
    FeS_L1 = Cofactor("FeS_L1", [-0.701])
    FeS_L2 = Cofactor("FeS_L2", [-0.529])
    FeS_H1 = Cofactor("FeS_H1", [mu_FeS_H1])
    S_FAD = Cofactor("S_FAD", [-0.3005, -0.2515])

    net.addCofactor(L_FAD)
    net.addCofactor(FeS_L1)
    net.addCofactor(FeS_L2)
    net.addCofactor(FeS_H1)
    net.addCofactor(S_FAD)

    # Low-potential branch
    net.addConnection(L_FAD, FeS_L1, LP_length)   #B-L1
    net.addConnection(FeS_L1, FeS_L2, 9.6)   #L1-L2
    net.addConnection(L_FAD, FeS_L2, LP_length+9.6)     #B-L2
    # High-potential branch
    net.addConnection(L_FAD, FeS_H1, HP_length)    #B-H1
    net.addConnection(FeS_H1, S_FAD, 9.6)     #H1-H2
    net.addConnection(L_FAD, S_FAD, HP_length+9.6)     #B-H2
    # Other connections
    net.addConnection(FeS_L1, FeS_H1, LP_length+HP_length)    #L1-H1
    net.addConnection(FeS_L1, S_FAD, LP_length+HP_length+9.6)    #L1-H2
    net.addConnection(FeS_L2, FeS_H1, LP_length+HP_length+9.6)    #L2-H1
    net.addConnection(FeS_L2, S_FAD, LP_length+HP_length+9.6+9.6)    #L2-H2

    L_FAD_mid = (L_FAD.redox[0]+L_FAD.redox[1])/2
    S_FAD_mid = (S_FAD.redox[0]+S_FAD.redox[1])/2
    L_FAD_NADPH_reservoir_rate = NADPH_L_FAD_reservoir_rate * np.exp(net.beta*2*(mu_NADPH-L_FAD_mid))
    net.addReservoir("NADPH", L_FAD, 2, 2, 2*(L_FAD_mid-mu_NADPH), L_FAD_NADPH_reservoir_rate)
    net.addReservoir("Fd", FeS_L2, 1, 1, FeS_L2.redox[0]-mu_Fd, L2_Fdox_reservoir_rate)
    net.addReservoir("NAD", S_FAD, 2, 2, 2*(S_FAD_mid-mu_NAD), S_FAD_NAD_reservoir_rate)    # two-electron concerted step S-FAD -> NAD+

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1
    t = 10 ** 4 # longer time to reach SS
    pop_MEK = net.evolve(t, pop_MEK_init)

    D_flux = net.getReservoirFlux("NADPH", pop_MEK)
    H_flux = net.getReservoirFlux("NAD", pop_MEK)
    L_flux = net.getReservoirFlux("Fd", pop_MEK)

    # short circuit pathways
    D1_to_H1 = net.getCofactorFlux(L_FAD, 1, FeS_H1, 1, pop_MEK)
    L1_to_D2 = net.getCofactorFlux(FeS_L1, 1, L_FAD, 2, pop_MEK)

    # compute bifurcation metrics
    F_sc = D1_to_H1 + L1_to_D2 # inverse of sum of short circuit fluxes
    F_yield = abs(D_flux) + abs(H_flux) + abs(L_flux)

    F = alpha * F_sc - (1-alpha) * F_yield

    return F, F_sc, F_yield, D_flux, H_flux, L_flux, D1_to_H1, L1_to_D2

def make_wrapped_obj(alpha):
    '''
    wrapper function for obj_func_full
    '''
    def wrapped_obj(potentials):
        results = []
        for i in range(potentials.shape[0]):
            pH1_array = [[potentials[i, 0]]]  # shape (1,1) to match original func
            F, *_ = obj_func_full(pH1_array, alpha)
            results.append([F])  # must be a list of lists
        return np.array(results)
    return wrapped_obj

def run_single_job(alpha):
    '''
    Runs t serial runs of the Bayesian Optimizer and returns the best optimization results
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
    alphas = []
    
    # need to change the iteration count based on t test
    for t in range(100):
        seed(t * 100 + 500)
        bounds = [{'name': 'H1', 'type': 'continuous', 'domain': (-0.160, 0.240)}]

        maxiter = 30
        wrapped = make_wrapped_obj(alpha)

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
        F_val, F_sc_val, F_yield_val, fluxD, fluxHR, fluxLR, D_H1_flux, L1_D_flux = obj_func_full(best_potentials, alpha)

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
        alphas.append(alpha)

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
    alphas_best = alphas[min_idx]

    return [alphas_best, F_t, F_sc_best, F_yield_best, fluxD_best, fluxHR_best, fluxLR_best, best_potentials[0][0], sc1_best, sc2_best]

if __name__ == '__main__':
    t_start = time.time()
    timestr = time.strftime("%Y%m%d")
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    alphas = np.linspace(0, 1, 3)
    chunk = np.array_split(alphas, num_tasks)[task_id]
    results = [run_single_job(alpha) for alpha in chunk]

    # save data
    columns = ["alpha", "F", "F_sc", "F_yield", "fluxD", "fluxHR", "fluxLR", "potential_H1", "D_H1_flux", "L1_D_flux"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"Nfn1alphas_{task_id}_"+timestr+".csv", index=False)
    
    t_end = time.time()
    runtime = t_end - t_start
    print(f"Total runtime: {runtime:.2f} seconds")
    

