'''
MEK.py
Date: June 13, 2025
Author: Emily Wang
Nfn-1 modeled with infinite reservoir MEK
'''
from MEK_public import *

import numpy as np
from numpy.linalg import eig

#packages for plotting
import matplotlib.pyplot as plt

data_points = 100
min_time = 0
max_time = 5*10**(1)
time = np.linspace(min_time, max_time, data_points)

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

def Nfn1(mu_FeS_H1, t):
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
    pop_MEK = net.evolve(t, pop_MEK_init)

    D_flux = net.getReservoirFlux("NADPH", pop_MEK)
    H_flux = net.getReservoirFlux("NAD", pop_MEK)
    L_flux = net.getReservoirFlux("Fd", pop_MEK)

    # pathway fluxes
    # normal pathways 
    D1_to_L1 = net.getCofactorFlux(L_FAD, 1, FeS_L1, 1, pop_MEK) # D1 = singly reduced before ET / D for LPB
    L1_to_L2 = net.getCofactorFlux(FeS_L1, 1, FeS_L2, 1, pop_MEK)
    D2_to_H1 = net.getCofactorFlux(L_FAD, 2, FeS_H1, 1, pop_MEK) # D2 = doubly reduced before ET / D for HPB
    H1_to_SFAD_1 = net.getCofactorFlux(FeS_H1, 1, S_FAD, 1, pop_MEK) # S-FAD can be doubly reduced
    H1_to_SFAD_2 = net.getCofactorFlux(FeS_H1, 1, S_FAD, 2, pop_MEK)

    # short circuit pathways
    D1_to_H1 = net.getCofactorFlux(L_FAD, 1, FeS_H1, 1, pop_MEK)
    L1_to_D2 = net.getCofactorFlux(FeS_L1, 1, L_FAD, 2, pop_MEK)

    # ratio_HD = H_flux/D_flux
    # ratio_LD = L_flux/D_flux

    # print("mu_FeS_H1=", mu_FeS_H1)
    print("------","t=",t,"-------")
    print("D_flux=", D_flux, "H_flux=", H_flux, "L_flux=", L_flux)
    print("-----normal pathways-----")
    print("D1_to_L1_flux=", D1_to_L1, "L1_to_L2_flux=", L1_to_L2, "D2_to_H1_flux=", D2_to_H1, "H1_to_SFAD1_flux=", H1_to_SFAD_1, "H1_to_SFAD2_flux=", H1_to_SFAD_2)
    print("-----short circuit pathways-----")
    print("D1_to_H1_flux=", D1_to_H1, "L1_to_D2_flux=", L1_to_D2)
    # print("sum of flux=", D_flux+H_flux+L_flux)

    # return ratio_HD, ratio_LD
    return D_flux, H_flux, L_flux, D1_to_L1, L1_to_L2, D2_to_H1, H1_to_SFAD_1, H1_to_SFAD_2, D1_to_H1, L1_to_D2

def metrics(fluxD, fluxHR, fluxLR):
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
    print("F_slip=", F_slip, "F_yield=", F_yield)

    return F_slip, F_yield

N = 100
ztime = 10**(-5)
# ztime = 0.000006   ## Try plotting for longer time
dt = 9/(N-1)

time = ztime*(10**(N*dt))
print(time)
print("==== with bump ====")
Nfn1(0.08, time)
#F_slip, F_yield = metrics(NADPH, NAD, Fd)

print("==== ramp ====")
Nfn1(-0.118, time)
#F_slip, F_yield = metrics(NADPH, NAD, Fd)