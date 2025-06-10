# sandbox.py
# for testing

import numpy as np
import pandas as pd

bump_0 = pd.read_csv('BestBump_alpha0_20250606.csv')
bump_1 = pd.read_csv('BestBump_alpha1_20250609.csv')

sorted_y = bump_1['F_yield'].sort_values()
sorted_s = bump_1['F_slip'].sort_values()

print('F_yield: min, max - 1, max:', sorted_y.iloc[0], sorted_y.iloc[-2], sorted_y.iloc[-1])
print('F_slip: min, max - 1, max:', sorted_s.iloc[0], sorted_s.iloc[-2], sorted_s.iloc[-1])

F_yield_0 = bump_0["F_yield"]
F_yield_0 = F_yield_0[F_yield_0 != F_yield_0.max()]

F_yield_1 = bump_1["F_yield"]
F_yield_1 = F_yield_1[F_yield_1 != F_yield_1.max()]

F_slip_diff = bump_1['F_slip'] - bump_0['F_slip']
F_yield_diff = F_yield_1 - F_yield_0

print('F_slip diff (min, max): ', min(F_slip_diff), max(F_slip_diff))
print('F_yield diff (min, max): ', min(F_yield_diff), max(F_yield_diff))