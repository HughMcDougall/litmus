import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob

TARGS = glob("*.pckl")

bad_targs = []
for i, targ in enumerate(TARGS):
    with open(targ, "rb") as f:
        data = pickle.load(f)
    if np.isnan(data["runtime"]):
        print("Run %i is bad" % i)
        bad_targs.append(data)

print("out of %i runs, %i have failed" % (len(TARGS), len(bad_targs)))

for targ in bad_targs:
    d1, d2 = targ["mock_data"]

    T1, Y1, E1 = d1.T
    T2, Y2, E2 = d2.T

    plt.figure()
    plt.errorbar(T1, Y1, E1, fmt='none', capsize=2)
    plt.errorbar(T2, Y2, E2, fmt='none', capsize=2)
    plt.show()
