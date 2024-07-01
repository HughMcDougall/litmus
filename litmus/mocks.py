'''
Some handy sets of mock data for use in testing

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
from copy import deepcopy as copy
import matplotlib.pyplot as plt

import types


# ===================================================

def mock_cadence(maxtime, seed=0, cadence=7, cadence_var=1, season=180, season_var=14, N=1024):
    '''
    Returns time series X values for a mock signal
    :param maxtime:
    :param cadence:
    :param cadence_var:
    :param season:
    :param season_var:
    :param N:

    returns as array of sample times
    '''

    np.random.seed(seed)

    # Generate Measurements
    diffs = np.random.randn(N) * cadence_var / np.sqrt(2) + cadence
    X = np.cumsum(diffs)
    X = X[np.where((X < (maxtime * 2)))[0]]
    X += np.random.randn(len(X)) * cadence_var / np.sqrt(2)

    # Make windowing function
    if season is not None and season != 0:

        no_seasons = maxtime // season
        window = np.zeros(len(X))
        for n in range(no_seasons):
            if n % 2 == 0: continue
            tick = np.tanh((X - season * n) / season_var)
            tick -= np.tanh((X - season * (n + 1)) / season_var)
            tick /= 2
            window += tick

        R = np.random.rand(len(window))

        X = X[np.where((R < window) * (X < maxtime))[0]]
    else:
        X = X[np.where(X < maxtime)[0]]

    return (X)


def subsample(X, Y, Xsample):
    '''
    Linearly interpolates between X and Y and returns at positions Xsample
    '''
    out = np.interp(Xsample, X, Y)
    return (out)


def outly(Y, q):
    '''
    outly(Y,q):
    Returns a copy of Y with fraction 'q' elements replaced with
    normally distributed outliers
    '''
    I = np.random.rand(len(Y)) < q
    Y[I] = np.random.randn() * len(I)
    return (Y)


def gp_realization(X, err=0.0, tau=400.0,
                   basekernel=tinygp.kernels.quasisep.Exp,
                   seed=None):
    '''
    Generates a gaussian process at times X and errors err

    :param X:
    :param errmag:
    :param tau:
    :param basekernel:
    :param X_true:
    :param seed:

    Returns as lightcurve object
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

    gp = GaussianProcess(basekernel(scale=tau), X)
    Y = gp.sample(jax.random.PRNGKey(seed))

    return (lightcurve(T=X, Y=Y, E=E))


# ================================================

class mock(object):
    def __init__(self, seed=0, **kwargs):
        defaultkwargs = {'tau': 400.0,
                         'cadence': [7, 30],
                         'cadence_var': [1, 5],
                         'season': 180,
                         'season_var': 14,
                         'N': 2048,
                         'maxtime': 360 * 5,
                         'lag': 30,
                         'E': [0.01, 0.1],
                         'E_var': [0.0, 0.0]
                         }

        self.lc, self.lc_1, self.lc_2 = None, None, None
        self.lag = 0.0
        kwargs = defaultkwargs | kwargs

        for key in ['cadence', 'cadence_var', 'E', 'E_var']:
            if not (isiter(kwargs[key])): kwargs[key] = [kwargs[key], kwargs[key]]
        for key, var in zip(kwargs.keys(), kwargs.values()):
            self.__setattr__(key, var)

        self.generate(seed=seed)
        return

    def __call__(self, seed=0, **kwargs):
        self.generate(seed=seed)
        return (copy(self))

    def generate_true(self, seed=0):
        '''
        Generates an underlying true DRW signal and stores in the self attribute self.lc
        :param seed:
        :return:
        '''
        X = np.linspace(0.0, self.maxtime + self.lag * 2, self.N)
        Y = gp_realization(X, tau=self.tau, seed=seed).Y
        self.lc = lightcurve(X, Y).trim(Tmin=0, Tmax=self.maxtime)
        return (X, Y)

    def generate(self, seed=0):
        '''
        Generates a mock and sampled light-curve including a delayed response and stores in the self-attributes
        self.lc_1 and self.lc_2
        :param seed:
        :return:
        '''
        X, Y = self.generate_true(seed=seed)

        X1 = mock_cadence(self.maxtime, seed, cadence=self.cadence[0], cadence_var=self.cadence_var[0],
                          season=self.season, season_var=self.season_var,
                          N=self.N)
        X2 = mock_cadence(self.maxtime, seed, cadence=self.cadence[1], cadence_var=self.cadence_var[1],
                          season=self.season, season_var=self.season_var,
                          N=self.N)

        Y1, Y2 = subsample(X, Y, X1), subsample(X - self.lag, Y, X2)
        E1, E2 = [np.random.randn(len(x)) * ev + e for x, ev, e in zip([X1, X2], self.E_var, self.E)]

        Y1 += np.random.randn(len(X1)) * abs(E1)
        Y2 += np.random.randn(len(X2)) * abs(E2)

        self.lc_1 = lightcurve(X1, Y1, E1)
        self.lc_2 = lightcurve(X2, Y2, E2)

        return

    def plot(self, axis=None, true_args={}, series_args={}):
        if axis is None:
            plt.figure()
            axis = plt.gca()
        true_args |= {'lw': 1, 'c': 'k', 'alpha': 0.5, 'label': 'True Signal', 'lw': 1}
        series_args |= {'lw': 1, 'c1': 'tab:blue', 'c2': 'tab:orange', 'alpha': 1.0, 'capsize': 2, 'lw': 3}
        axis.plot(self.lc.T, self.lc.Y, **true_args)
        axis.plot(self.lc.T - self.lag, self.lc.Y, **true_args | {'c': 'tab:red', 'alpha': 0.5})

        series_args1, series_args2 = copy(series_args) | {'c': series_args['c1']}, copy(series_args) | {
            'c': series_args['c2']}
        series_args1.pop('c1'), series_args2.pop('c1')
        series_args1.pop('c2'), series_args2.pop('c2')
        axis.errorbar(self.lc_1.T, self.lc_1.Y, self.lc_1.E, fmt='none',
                      label="Time Series 1",
                      **series_args1
                      )
        axis.errorbar(self.lc_2.T, self.lc_2.Y, self.lc_2.E, fmt='none',
                      label="Time Series 2",
                      **series_args2
                      )


# ================================================

# ================================================
# CASE A - WELL OBSERVED SMOOTH CURVES


def determ_gen(self, seed=0):
    f = lambda x: np.exp(-((x - 1000) / 2 / (64)) ** 2 / 2)
    X = np.linspace(0.0, self.maxtime + self.lag * 2, self.N)
    Y = f(X)
    self.lc = lightcurve(X, Y).trim(Tmin=0, Tmax=self.maxtime)
    return (X, Y)


# Change the way mock A generates a time series
mock_A = mock(season=None, lag=300)
mock_A.generate_true = types.MethodType(determ_gen, mock_A)
mock_A()

mock_A_01, mock_A_02, lag_A = mock_A.lc_1, mock_A.lc_2, mock_A.lag

# ================================================
# CASE B - SEASONAL GP
mock_B = mock(lag=256, maxtime=360 * 5, E=[0.01, 0.01], seed=1, season=180)
mock_B_00 = mock_B.lc
mock_B_01 = mock_B.lc_1
mock_B_02 = mock_B.lc_2
lag_B = mock_B.lag

# ================================================
# CASE C - UN-SEASONAL GP
mock_C = mock(lag=128, maxtime=360 * 5, E=[0.01, 0.01], season=None)

mock_C_00 = mock_C.lc
mock_C_01 = mock_C.lc_1
mock_C_02 = mock_C.lc_2
lag_C = mock_C.lag

# ================================================
if __name__ == "__main__":
    for x in mock_A, mock_B, mock_C:
        plt.figure()
        plt.title("Seasonal GP, lag = %.2f" % x.lag)

        x.plot(axis=plt.gca())

        plt.legend()
        plt.xlabel("Time (Days)")
        plt.ylabel("Signal Strength")
        plt.axhline(0.0, ls='--', c='k', zorder=-10)
        plt.axhline(1.0, ls=':', c='k', zorder=-10)
        plt.axhline(-1.0, ls=':', c='k', zorder=-10)

        plt.grid()

        plt.show()
