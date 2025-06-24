# sandbox.py
# for testing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Nfn1_bump_size_20250623.csv')
plt.scatter(df['pH1'], df['F_slip'], s=20)
plt.title('F_slip vs. potential on H1', fontsize=10)
plt.xlabel('Potential on H1 (eV)')
plt.ylabel('F_slip')
plt.tight_layout()
plt.legend()
plt.show()

