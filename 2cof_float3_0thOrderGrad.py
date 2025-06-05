'''
2cof_float3_0thOrderGrad
Author: Emily Wang
April 22, 2025
Gradient-based optimization method using Nelder-Mead method, 0th order optimization
'''

# === Import Modules ===
from Fncts_newresrate import *
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting

# === Full Objective Function ===
def obj_func(potentials, alpha):
    '''
    Run MEK simulation and compute the objective function.
    
    Args:
        potentials (list): Potentials [L2, D, H2] to be optimized.
        alpha (float): Weighting between slippage and yield.
    
    Returns:
        F (float): Weighted objective function.
        alpha (float)
        F_slip (float)
        F_yield (float)
        fluxD (float)
        fluxHR (float)
        fluxLR (float)
    '''
    # Fixed potentials
    pL1 = -0.25
    pH1 = 0.25

    # Extract floating potentials
    pL2 = potentials[0]
    pD1 = potentials[1]
    pD2 = -pD1
    pH2 = potentials[2]

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

    # Normalize fluxes
    fluxes = np.array([fluxD, fluxHR, fluxLR])
    max_flux = np.max(np.abs(fluxes))
    fluxD_norm = fluxD / abs(max_flux)
    fluxHR_norm = fluxHR / abs(max_flux)
    fluxLR_norm = fluxLR / abs(max_flux)

    # Efficiency metric: slippage
    F_slip = math.sqrt((fluxD_norm + 1)**2 + (fluxHR_norm - 0.5)**2 + (fluxLR_norm - 0.5)**2)

    # Bifurcation metric: yield
    F_yield = 1 / (abs(fluxD) + abs(fluxHR) + abs(fluxLR))

    # Combined objective
    # Third term adds a small curvature element
    F = alpha * F_slip + (1 - alpha) * F_yield

    return F, alpha, F_slip, F_yield, fluxD, fluxHR, fluxLR

# === Wrapped Objective for Optimizer ===
def make_wrapped_obj(alpha):
    def wrapped_obj(potentials):
        F, *_ = obj_func(potentials, alpha)
        return F
    return wrapped_obj

# === Optimization Routine ===
def optimization(bounds, alpha, x0):
    wrapped = make_wrapped_obj(alpha)
    result = minimize(wrapped, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30, 'ftol': 1e-7, 'gtol': 1e-5})

    best_potentials = result.x
    F, alpha_out, F_slip, F_yield, fluxD, fluxHR, fluxLR = obj_func(best_potentials, alpha)

    return {
        'alpha': alpha_out,
        'F': F,
        'F_slip': F_slip,
        'F_yield': F_yield,
        'abs(fluxD)': abs(fluxD),
        'fluxHR': fluxHR,
        'fluxLR': fluxLR,
        'potential_D1': best_potentials[1],
        'potential_L2': best_potentials[0],
        'potential_H2': best_potentials[2],
        'success': result.success,
        'message': result.message,
    }

def random_x0(bounds):
    return np.array([np.random.uniform(low, high) for (low, high) in bounds])

# === Main Block ===
if __name__ == '__main__':
    '''
    t_start = time.time()
    date = time.strftime("%Y%m%d")

    alphas = np.linspace(0, 1, 2)
    bounds = [(-0.400, -0.05), (-0.500, -0.300), (0.05, 0.400)]  # L2, D, H2

    results_best = []
    for alpha in alphas:
        results = []
        for i in range(5):
            x0 = random_x0(bounds)
            result = optimization(bounds, alpha, x0)
            #results.append(result)
        #best_index, best_result = min(enumerate(results), key=lambda r: r[1]['F'])
        #print(best_index)
            result["point"]=str(x0)
            results_best.append(result)

    # Save results
    columns = ["point","alpha", "F", "F_slip", "F_yield", 
               "abs(fluxD)", "fluxHR", "fluxLR", 
               "potential_D1", "potential_L2", "potential_H2",
               "success", "message"]
    df = pd.DataFrame(results_best, columns=columns)
    df.to_csv(f"2cof_float3_grad_{date}.csv", index=False)

    print("Runtime (s):", time.time() - t_start)
    '''

    t_start = time.time()
    bounds = [(-0.400, -0.05), (-0.500, -0.300), (0.05, 0.400)]  # L2, D, H2
    results_best = []
    results = []
    for i in range(20):
        x0 = random_x0(bounds)
        result = optimization(bounds, 0, x0)
        result["point"]=str(x0)
        results_best.append(result)
    
    # Save results
    columns = ["point","alpha", "F", "F_slip", "F_yield", 
               "abs(fluxD)", "fluxHR", "fluxLR", 
               "potential_D1", "potential_L2", "potential_H2",
               "success", "message"]
    df = pd.DataFrame(results_best, columns=columns)
    df.to_csv(f"2cof_float3_grad_{date}.csv", index=False)
    best_index, best_result = min(enumerate(results), key=lambda r: r[1]['F'])
    print(best_index)
    print(best_result)

    print("Runtime (s):", time.time() - t_start)



