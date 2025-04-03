'''
runs_serial.py
Author: Emily Wang
Feb 2025
Bayesian Optimization on simple toy bifurcator DHL from Kiriko's finite reservoir manuscript/Jon Yuly's paper
Testing how many iterations of the optimizer to run
Runs optimizations in series (for loop)
'''

#Import Modules
from Fncts_newresrate import *
import math

import numpy as np
from numpy.linalg import eig
from numpy.random import seed

import matplotlib.pyplot as plt

import GPyOpt
from GPyOpt.methods import BayesianOptimization

import pandas as pd
import time

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
    pL1 = -0.25
    pH1 = 0.25

    # extract floating potentials
    pL2 = potentials[0][0]
    pD1 = potentials[0][1] # first reduction potential (negative)
    pD2 = -pD1 # second reduction potential is set to negative of first (positive)
    pH2 = potentials[0][2]

    # run MEK - currently set to open conformation for open flow to low potential branch
    # fixed hyperparameters
    res_rate = 10**5 # rate for cofactor --> reservoir ET
    res_rate_red = 2*10**5 # increased rate for enhanced binding affinity of NAD when B2/C1 reduced
        
    # Time step variables
    N = 20 # time steps
    ztime = 0.00001 # intial time
    dt = 7/(N-1) # controls the orders of magntidue explored (7 in this case)

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

def wrapped_obj_func(potentials):
    '''
    wrapped function for optimizer
    '''
    output = obj_func_weighted(potentials, alpha = alpha)
    #output = obj_func_weighted(potentials, alpha = user_alpha) # use this for user input alpha
    return output

t_start = time.time()

# set fixed alpha 
alpha = 0.5

# store data for plotting
F_output = [] # output F for each of the t trials
F_min = [] # t-th index is the minimum F of the first t trials
FluxD = []
FluxHR = []
FluxLR = []
FluxD_best = []
FluxHR_best = []
FluxLR_best = []
T = []
Results = []

for t in range(200):
    seed(t * 100 + 500)
    bounds = [
        {'name': 'L2', 'type': 'continuous', 'domain': (-0.200, 0)},
        {'name': 'D1', 'type': 'continuous', 'domain': (-0.500, -0.300)},
        {'name': 'H2', 'type': 'continuous', 'domain': (0, 0.200)}]

    maxiter = 30

    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f = wrapped_obj_func, 
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
    T.append(t) # iteration number
    F_output.append(F) # F for t-th iteration
    FluxD.append(abs(fluxD))
    FluxHR.append(fluxHR)
    FluxLR.append(fluxLR)

    # find best trial and save data
    min_idx, F_t = min(enumerate(F_output), key=lambda x: x[1])
    min_idx = int(min_idx)
    F_min.append(F_t) # min F of first t iterations
    fluxD_best = FluxD[min_idx]
    fluxHR_best = FluxHR[min_idx]
    fluxLR_best = FluxLR[min_idx]
    FluxD_best.append(fluxD_best)
    FluxHR_best.append(fluxHR_best)
    FluxLR_best.append(fluxLR_best)
    Results.append([t, F, F_t, abs(fluxD), fluxHR, fluxLR, fluxD_best, fluxHR_best, fluxLR_best]) # for saving in data frame

# save all output as csv
columns = ["t", "F_t", "F_(t)", "abs(fluxD)", "fluxHR", "fluxLR", "abs(fluxD)_(t)", "fluxHR_(t)", "fluxLR_(t)"]
df = pd.DataFrame(Results, columns=columns)
df.to_csv("DHLOpt_runs_serial_20250217.csv", index=False)

t_end = time.time()
runtime = t_end - t_start
print("runtime: ", runtime)

# Plot of F_(t) vs. t
plt.figure(figsize=(8, 5))
plt.plot(T, F_min, linestyle='-', color='blue')
plt.xlabel('t (runs)')
plt.ylabel('Minimum F of first t runs')
plt.title('Optimization result when F is the minimum of t optimizations')
plt.legend()
plt.grid(True)
plt.savefig('F(t)_vs_t.png')

# plot of best fluxes vs. t
plt.figure(figsize=(8, 5))
plt.plot(T, fluxD_best, label='Best flux DR', marker='^', linestyle='-', color='green')
plt.plot(T, fluxHR_best, label='Best flux HR', marker='o', linestyle='-', color='blue')
plt.plot(T, fluxLR_best, label='Best flux LR', marker='s', linestyle='-', color='red')
plt.xlabel('t (runs)')
plt.ylabel('Flux')
plt.title('Best fluxes vs t')
plt.legend()
plt.grid(True)
plt.savefig('BestFluxes_vs_t.png')