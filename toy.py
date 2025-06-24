'''
toy.py
Date: June 4, 2025
Author: Emily Wang
MEK set up for toy model
'''
from MEK_public import *
import math
import numpy as np

def toy(pH1, slopeL, slopeH):
    pL1 = -0.4755 + slopeL
    pL2 = -0.4755 + 2*slopeL
    pH1 = pH1
    pH2 = 0.4755 + 2*slopeH

    # Hyperparameters
    res_rate = 50  # Ortiz JBC 2023
    N = 20
    ztime = 0.00001
    dt = 7 / (N - 1)

    net = Network()

    # Cofactor definitions
    L1 = Cofactor("L1", [pL1])
    L2 = Cofactor("L2", [pL2])
    D = Cofactor("D", [-0.4755, 0.4755])
    H1 = Cofactor("H1", [pH1])
    H2 = Cofactor("H2", [pH2])

    # Add cofactors to network
    for cofactor in [L1, L2, D, H1, H2]:
        net.addCofactor(cofactor)

    # Add connections (distances in angstroms)
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

    # Add reservoirs
    net.addReservoir("DR", D, 2, 2, -0.15, res_rate)
    net.addReservoir("LR", L2, 1, 1, -0.109, res_rate)
    net.addReservoir("HR", H2, 1, 1, 0, res_rate)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # Initial condition: no electrons
    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1


    # Simulate evolution
    time_val = ztime * (10 ** (N * dt))
    pop_MEK = net.evolve(time_val, pop_MEK_init)

    # Compute fluxes
    fluxD = net.getReservoirFlux("DR", pop_MEK)
    fluxHR = net.getReservoirFlux("HR", pop_MEK)
    fluxLR = net.getReservoirFlux("LR", pop_MEK)

    fluxes = np.array([fluxD, fluxHR, fluxLR])
    max_flux = np.max(np.absolute(fluxes))
    if max_flux == 0:
        fluxD_norm = 0
        fluxHR_norm = 0
        fluxLR_norm = 0
        F_slip = 0
        F_yield = 0
    else:
        fluxD_norm = fluxD / abs(max_flux)
        fluxHR_norm = fluxHR / abs(max_flux)
        fluxLR_norm = fluxLR / abs(max_flux)
        F_slip = math.sqrt((fluxD_norm + 1) ** 2 + (fluxHR_norm - 0.5) ** 2 + (fluxLR_norm - 0.5) ** 2)
        F_yield = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

        F_slip = math.sqrt((fluxD_norm + 1) ** 2 + (fluxHR_norm - 0.5) ** 2 + (fluxLR_norm - 0.5) ** 2)
        F_yield = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))


    return fluxD, fluxHR, fluxLR, F_slip, F_yield

if __name__ == '__main__':
    slopeL = 0.191
    slopeH = -0.158
    pH1_bump = 0.4755 + slopeH + 0.198
    pH1_ramp = 0.4755 + slopeH
    fluxD, fluxHR, fluxLR, F_slip, F_yield = toy(pH1_bump, slopeL, slopeH)
    print(fluxD, fluxHR, fluxLR, F_slip, F_yield)

    fluxD, fluxHR, fluxLR, F_slip, F_yield = toy(pH1_ramp, slopeL, slopeH)
    print(fluxD, fluxHR, fluxLR, F_slip, F_yield)
