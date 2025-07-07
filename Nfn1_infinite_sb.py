'''
Nfn1_infinite_sb.py
Author: Kiriko Terai
Date: June 17, 2025
Nfn-1 infinite reservoir model from Kiriko 
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

NADPH_flux = []
NAD_flux = []
Fd_flux = []
time_list = []

N = 50
ztime = 10**(-2)
# ztime = 0.000006   ## Try plotting for longer time
dt = 9/(N-1)

for n in range(N):
    if n == 0:
        time = 0.0
        time_list.append(time)
        NADPH, NAD, Fd = Nfn1(0.08, time)
        NADPH_flux.append(NADPH)
        NAD_flux.append(NAD)
        Fd_flux.append(Fd)
    else:
        time = ztime*(10**(n*dt))
        time_list.append(time)
        NADPH, NAD, Fd = Nfn1(0.08, time)
        NADPH_flux.append(NADPH)
        NAD_flux.append(NAD)
        Fd_flux.append(Fd)

filtered_time_list = [t for t in time_list if t > 0.17]
truncate_len = len(time_list) - len(filtered_time_list)

fig = plt.figure()
plt.plot(filtered_time_list, NADPH_flux[truncate_len:], label="$J_{NADPH}$", color="darkviolet")
plt.plot(filtered_time_list, NAD_flux[truncate_len:], label="$J_{NAD^{+}}$", color="blue")
plt.plot(filtered_time_list, Fd_flux[truncate_len:], label="$J_{Fd_{ox}}$", color="red")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.xlabel("Time (s)", size="x-large")
plt.xscale("log")
plt.legend()
# plt.show()
fig.savefig("Nfn1_flux.pdf")
