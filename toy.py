'''
toy.py
Date: June 4, 2025
Author: Emily Wang
MEK set up for toy model
'''
from MEK_public import *
import math
import numpy as np

def toy(pH1, slopeL, slopeH, time_val):
    pL1 = -0.4755 + slopeL
    pL2 = -0.4755 + 2*slopeL
    pH1 = pH1
    pH2 = 0.4755 + 2*slopeH

    res_rate = 50

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
    net.addReservoir("DR", D, 2, 2, -0.072, res_rate)
    net.addReservoir("LR", L2, 1, 1, -0.109, res_rate)
    net.addReservoir("HR", H2, 1, 1, 0.004, res_rate)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # Initial condition: no electrons
    pop_MEK_init = np.zeros(net.adj_num_state)
    pop_MEK_init[0] = 1


    # Simulate evolution
    pop_MEK = net.evolve(time_val, pop_MEK_init)

    # Compute fluxes
    fluxD = net.getReservoirFlux("DR", pop_MEK)
    fluxHR = net.getReservoirFlux("HR", pop_MEK)
    fluxLR = net.getReservoirFlux("LR", pop_MEK)

    # short circuit pathway fluxes
    D1_to_H1 = net.getCofactorFlux(D, 1, H1, 1, pop_MEK)
    L1_to_D2 = net.getCofactorFlux(L1, 1, D, 2, pop_MEK)

    # compute bifurcation metrics
    F_sc = D1_to_H1 + L1_to_D2 # inverse of sum of short circuit fluxes
    F_yield = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

    return fluxD, fluxHR, fluxLR, F_sc, F_yield

if __name__ == '__main__':
    slopeL = 0.192
    slopeH = -0.160
    pH1_bump = 0.4755 + slopeH + 0.198
    pH1_ramp = 0.4755 + slopeH
    fluxD, fluxHR, fluxLR, F_sc, F_yield = toy(pH1_bump, slopeL, slopeH, 10**4)
    print(fluxD, fluxHR, fluxLR, F_sc, F_yield)

    fluxD, fluxHR, fluxLR, F_sc, F_yield = toy(pH1_ramp, slopeL, slopeH, 10**4)
    print(fluxD, fluxHR, fluxLR, F_sc, F_yield)

    '''
    fluxD = []
    fluxH = []
    fluxL = []
    time_list = []

    N = 50
    ztime = 10**(-5)
    # ztime = 0.000006   ## Try plotting for longer time
    dt = 9/(N-1)

    for n in range(N):
        if n == 0:
            time = 0.0
            time_list.append(time)
            D, H, L = toy(pH1_bump, slopeL, slopeH, time)
            fluxD.append(D)
            fluxH.append(H)
            fluxL.append(L)
        else:
            time = ztime*(10**(n*dt))
            time_list.append(time)
            D, H, L = toy(pH1_bump, slopeL, slopeH, time)
            fluxD.append(D)
            fluxH.append(H)
            fluxL.append(L)

    fig = plt.figure()
    plt.plot(time_list, fluxD, label="$J_{D}$", color="darkviolet")
    plt.plot(time_list, fluxH, label="$J_{H}$", color="blue")
    plt.plot(time_list, fluxL, label="$J_{L}$", color="red")
    plt.ylabel("Flux (s$^{-1}$)", size="x-large")
    plt.xlabel("Time (s)", size="x-large")
    plt.xscale("log")
    plt.legend()
    # plt.show()
    fig.savefig("toy_flux.pdf")
    '''