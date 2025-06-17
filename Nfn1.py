'''
MEK.py
Date: June 13, 2025
Author: Emily Wang
Nfn-1 modeled with infinite reservoir MEK
'''
from Fncts_newresrate import *
import math
import numpy as np

def run_MEK(potentials):
    # Fixed potentials
    pL1 = -0.701
    pL2 = -0.529
    pH1 = 0.08
    pH2 = -0.200
    pD1 = -0.911 # inverted potentials?
    pD2 = 0.039

    # Hyperparameters
    res_rate = 50  # Ortiz JBC 2023
    N = 20
    ztime = 0.00001
    dt = 7 / (N - 1)

    net = Network()

    # Cofactor definitions
    L1 = Cofactor("L1", [pL1])
    L2 = Cofactor("L2", [pL2])
    D = Cofactor("D", [pD1, pD2])
    H1 = Cofactor("H1", [pH1])
    H2 = Cofactor("H2", [pH2])

    # Add cofactors to network
    for cofactor in [L1, L2, D, H1, H2]:
        net.addCofactor(cofactor)

    # Add connections (distances in angstroms)
    net.addConnection(D, L1, 7.5)
    net.addConnection(L1, L2, 9.6)
    net.addConnection(D, H1, 14.1)
    net.addConnection(H1, H2, 8.6)
    net.addConnection(D, L2, 20)
    net.addConnection(D, H2, 20)
    net.addConnection(L1, H1, 20)
    net.addConnection(L1, H2, 30)
    net.addConnection(L2, H1, 30)
    net.addConnection(L2, H2, 40)

    # Add reservoirs
    net.addReservoir("DR", D, 2, 2, 0.20, res_rate)
    net.addReservoir("LR", L2, 1, 1, 0, res_rate)
    net.addReservoir("HR", H2, 1, 1, 0, res_rate)

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # Initial condition: no electrons
    pop_MEK_init = np.zeros(net.num_state)
    pop_MEK_init[0] = 1

    # Simulate evolution
    time_val = ztime * (10 ** (N * dt))
    pop_MEK = net.evolve(time_val, pop_MEK_init)

    # Compute fluxes
    fluxD = net.getReservoirFlux("DR", pop_MEK)
    fluxHR = net.getReservoirFlux("HR", pop_MEK)
    fluxLR = net.getReservoirFlux("LR", pop_MEK)

    return fluxD, fluxHR, fluxLR

if __name__ == '__main__':
    fluxes = run_MEK([-0.05, -0.4577413390118908, 0.3996580526095928])
    print(fluxes)