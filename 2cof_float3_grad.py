'''
2cof_float3_grad
Author: Emily Wang
April 22, 2025
Gradient based optimization method for 2cof_float3 simulation.
'''
# Import Modules
# modules for MEK
from Fncts_newresrate import *
import math

import numpy as np
from numpy.linalg import eig
from numpy.random import seed
import scipy.optimize

# for parallelizing
from multiprocessing import Pool, cpu_count

# standard packages
import matplotlib.pyplot as plt
import pandas as pd
import time

import os

def obj_func(potentials, alpha):
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
    pL2 = potentials[0]
    pD1 = potentials[1] # first reduction potential (negative)
    pD2 = -pD1 # second reduction potential is set to negative of first (positive)
    pH2 = potentials[2]

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

    F = alpha * F_slip + (1 - alpha) * F_yield

    return alpha, F, F_slip, F_yield, fluxD, fluxHR, fluxLR

def optimization(alpha):
    obj_func_wrap = lambda x: obj_func(x, alpha)
    # set x0, initial guess before running!
    #x0 = []
    scipy.optimize.minimize(obj_func(x), , method='CG', maxiter=500)