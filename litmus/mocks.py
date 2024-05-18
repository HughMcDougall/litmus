import numpy as np
from lightcurve import lightcurve

#::::::::::::::::::::
# Mock Data
print("Making Mock Data")
f = lambda x: np.exp(-((x - 8) / 2) ** 2 / 2)

X1 = np.linspace(0, 2 * np.pi * 3, 64)
X2 = np.copy(X1)[::2]
X1 += np.random.randn(len(X1)) * X1.ptp() / (len(X1) - 1) * 0.25
X2 += np.random.randn(len(X2)) * X2.ptp() / (len(X2) - 1) * 0.25

E1, E2 = [np.random.poisson(10, size=len(X)) * 0.005 for i, X in enumerate([X1, X2])]
E2 *= 2

lag_true = np.pi
Y1 = f(X1) + np.random.randn(len(E1)) * E1
Y2 = f(X2 + lag_true) + np.random.randn(len(E2)) * E2

#::::::::::::::::::::
# Lightcurve object


data_1, data_2 = lightcurve(X1, Y1, E1), lightcurve(X2, Y2, E2)