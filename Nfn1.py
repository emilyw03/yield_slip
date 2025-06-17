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

    # ratio_HD = H_flux/D_flux
    # ratio_LD = L_flux/D_flux

    # print("mu_FeS_H1=", mu_FeS_H1)
    print("------","t=",t,"-------")
    print("D_flux=", D_flux, "H_flux=", H_flux, "L_flux=", L_flux)
    # print("sum of flux=", D_flux+H_flux+L_flux)

    # return ratio_HD, ratio_LD
    return D_flux, H_flux, L_flux

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

NADPH_flux = []
NAD_flux = []
Fd_flux = []
time_list = []

N = 100
ztime = 10**(-6)
# ztime = 0.000006   ## Try plotting for longer time
dt = 9/(N-1)

time = ztime*(10**(N*dt))
print("==== with bump ====")
NADPH, NAD, Fd = Nfn1(0.08, time)
F_slip, F_yield = metrics(NADPH, NAD, Fd)

print("==== ramp ====")
NADPH, NAD, Fd = Nfn1(-0.276, time)
F_slip, F_yield = metrics(NADPH, NAD, Fd)