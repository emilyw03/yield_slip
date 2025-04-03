# sandbox.py
# for testing

import numpy as np
from numpy.linalg import eig

import pandas as pd

import matplotlib.pyplot as plt

import GPyOpt
from GPyOpt.methods import BayesianOptimization

from multiprocessing import Pool, cpu_count

import time

def f(x):
    return x*x

if __name__ == '__main__':
    t_start = time.time()
    with Pool(processes=cpu_count()) as pool:         # start 4 worker processes
        print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"
    t_end = time.time()
    print('runtime: ', t_end - t_start)

    t_start = time.time()
    results = []
    for i in range(10):
        results.append(f(i))
    print(results)
    t_end = time.time()
    print('runtime: ', t_end - t_start)