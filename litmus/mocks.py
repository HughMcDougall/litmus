'''
Some handy sets of mock data for use in testing

_Extremely_ Rough placeholder. To be thoroughly rebuilt as a later date

HM Apr 2024
'''

# ============================================
# IMPORTS
import numpy as np
import pylab as pl

from lightcurve import lightcurve
import tinygp
from tinygp import GaussianProcess
from _utils import randint, isiter

import jax


# ===================================================

def mock_cadence(maxtime, cadence=7, err=1, seasonlength=180, N=1024):
    times = np.linspace(maxtime, N)
    diffs = np.random.randn(N) * err / np.sqrt(2) + cadence

    X = np.cumsum(diffs)
    window = np.where(np.sin(X * np.pi / (seasonlength)) > 0, 1, 0)

    X = X[np.where(window * (X < maxtime))[0]]
    X += np.random.randn(len(X)) * err / np.sqrt(2)

    return (X)


def mock_realization(X, err=0.0, tau=400.0,
                     basekernel=tinygp.kernels.quasisep.Exp, X_true=None,
                     seed=None):
    '''
    Generates mock realizations lightcurves for a GP of amplitude 1.0 and mean 0.0 for input times 'X' and error 'E'
    :param X:
    :param errmag:
    :param tau:
    :param basekernel:
    :param X_true:
    :param seed:
    :return:
    '''
    if seed is None: seed = randint()

    # -----------------
    # Generate errors
    N = len(X)
    if isiter(err):
        E = err
    else:
        E = np.random.randn(N) * np.sqrt(err) + err
    E = abs(E)

    # Combine and sort true and false
    if X_true is not None:
        X_all = np.array([*X, *X_true])
    else:
        X_all = X
    I = np.argsort(X_all)
    I_reverse = np.argsort(I)

    gp = GaussianProcess(basekernel(scale=tau), X_all[I])
    Y = gp.sample(jax.random.PRNGKey(seed))[I_reverse]

    if X_true is None:
        Y += np.random.randn(len(Y)) * E
        return lightcurve(T=X, Y=Y, E=E)
    else:
        Y1, Y2 = np.split(Y, [N])
        Y1 += np.random.randn(len(X)) * E
        return (lightcurve(T=X, Y=Y1, E=E),
                lightcurve(T=X_true, Y=Y2, E=np.zeros_like(X_true))
                )


# ============================================
# Mock Data

# ================================================
# CASE A - WELL OBSERVED SMOOTH CURVES

f = lambda x: np.exp(-((x - 8) / 2) ** 2 / 2)

X1 = np.linspace(0, 2 * np.pi * 3, 64)
X2 = np.copy(X1)[::2]
X1 += np.random.randn(len(X1)) * X1.ptp() / (len(X1) - 1) * 0.25
X2 += np.random.randn(len(X2)) * X2.ptp() / (len(X2) - 1) * 0.

E1, E2 = [np.random.poisson(10, size=len(X)) * 0.005 for i, X in enumerate([X1, X2])]
E2 *= 2

lag_true = np.pi
Y1 = f(X1) + np.random.randn(len(E1)) * E1
Y2 = f(X2 - lag_true) + np.random.randn(len(E2)) * E2

# Stretch to be t in [0,1000]
fac = 1000 / X1.max()
X1 *= fac
X2 *= fac
lag_true *= fac

#::::::::::::::::::::
# Lightcurve objects
mock_A_01, mock_A_02, lag_A = lightcurve(X1, Y1, E1), lightcurve(X2, Y2, E2), lag_true

# ================================================
# CASE B - SEASONAL GP
lag_B = 256
X = mock_cadence(maxtime=360 * 5)
X_true = np.linspace(X.min(), X.max(), 1024)
mock_B_01, mock_B_00 = mock_realization(X, X_true=X_true, err=0.01, seed=0, tau= 1000)

w = np.sin((mock_B_00.T + lag_B) * np.pi / 180) > 0
w = w.astype(float)
w /= w.sum()

I = np.random.choice(range(len(mock_B_00.T)), p=w, size=len(mock_B_01.T))
E = mock_B_01.E.copy()
np.random.shuffle(E)
mock_B_02 = lightcurve(T=mock_B_00.T[I] + lag_B, Y=mock_B_00.Y[I] + np.random.randn(len(E)) * E, E=E)
mock_B_02 = mock_B_02.trim(Tmin = mock_B_01.T.min(),
                           Tmax = mock_B_01.T.max())

# ================================================
# CASE C - UN-SEASONAL GP
lag_C = 128
X = mock_cadence(maxtime=360 * 5, seasonlength=360 * 5 * 2)
X_true = np.linspace(X.min(), X.max(), 1024)
mock_C_01, mock_C_00 = mock_realization(X, X_true=X_true, err=0.001, seed=1, tau= 1000)

I = np.random.choice(range(len(mock_C_00.T)), size=len(mock_C_01.T))
E = mock_C_01.E.copy()
np.random.shuffle(E)
mock_C_02 = lightcurve(T=mock_C_00.T[I] + lag_C, Y=mock_C_00.Y[I] + np.random.randn(len(E)) * E, E=E)
mock_C_02 = mock_C_02.trim(Tmin = mock_C_01.T.min(),
                           Tmax = mock_C_01.T.max())

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #---------
    plt.figure()

    plt.title("Smooth Curve, lag = %.2f" % lag_A)

    plt.errorbar(X1, Y1, E1, fmt='none', capsize=2)
    plt.errorbar(X2, Y2, E2, fmt='none', capsize=2, c='tab:orange')
    plt.grid()
    plt.legend(["Main Signal", "Delayed Signal"])
    plt.show()

    #---------
    plt.figure()

    plt.title("Seasonal GP, lag = %.2f" % lag_B)

    plt.errorbar(mock_B_01.T, mock_B_01.Y, mock_B_01.E, fmt='none', capsize=2, color='tab:blue')
    plt.errorbar(mock_B_02.T, mock_B_02.Y, mock_B_02.E, fmt='none', capsize=2, color='tab:red')
    plt.plot(mock_B_00.T, mock_B_00.Y, c='tab:orange', zorder=-1)

    plt.legend(['Observations', 'True Signal'])
    plt.xlabel("Time (Days)")
    plt.ylabel("Signal Strength")
    plt.axhline(0.0, ls='--', c='k', zorder=-10)
    plt.axhline(1.0, ls=':', c='k', zorder=-10)
    plt.axhline(-1.0, ls=':', c='k', zorder=-10)
    plt.grid()

    plt.show()

    #---------
    plt.figure()

    plt.title("Un-Seasonal GP, lag = %.2f" % lag_C)

    plt.errorbar(mock_C_01.T, mock_C_01.Y, mock_C_01.E, fmt='none', capsize=2, color='tab:blue')
    plt.errorbar(mock_C_02.T, mock_C_02.Y, mock_C_02.E, fmt='none', capsize=2, color='tab:red')
    plt.plot(mock_C_00.T, mock_C_00.Y, c='tab:orange', zorder=-1)

    plt.legend(['Observations', 'True Signal'])
    plt.xlabel("Time (Days)")
    plt.ylabel("Signal Strength")
    plt.axhline(0.0, ls='--', c='k', zorder=-10)
    plt.axhline(1.0, ls=':', c='k', zorder=-10)
    plt.axhline(-1.0, ls=':', c='k', zorder=-10)
    plt.grid()

    plt.show()
