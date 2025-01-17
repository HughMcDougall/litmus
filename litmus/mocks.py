'''
Some handy sets of mock data for use in testing

HM Apr 2024
'''

# ============================================
# IMPORTS

from copy import deepcopy as copy
import types

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

import jax

import tinygp
from tinygp import GaussianProcess

from litmus._utils import randint, isiter
from litmus.lightcurve import lightcurve


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
    T = np.cumsum(diffs)
    T = T[np.where((T < (maxtime * 2)))[0]]
    T += np.random.randn(len(T)) * cadence_var / np.sqrt(2)

    # Make windowing function
    if season is not None and season != 0:

        no_seasons = int(maxtime / season)
        window = np.zeros(len(T))
        for n in range(no_seasons):
            if n % 2 == 0: continue
            tick = np.tanh((T - season * n) / season_var)
            tick -= np.tanh((T - season * (n + 1)) / season_var)
            tick /= 2
            window += tick

        R = np.random.rand(len(window))

        T = T[np.where((R < window) * (T < maxtime))[0]]
    else:
        T = T[np.where(T < maxtime)[0]]

    return (T)


def subsample(T, Y, Tsample):
    '''
    Linearly interpolates between X and Y and returns at positions Xsample
    '''
    out = np.interp(Tsample, T, Y)
    return (out)


def outly(Y, q):
    '''
    outly(Y,q):
    Returns a copy of Y with fraction 'q' elements replaced with
    unit - normally distributed outliers
    '''
    I = np.random.rand(len(Y)) < q
    Y[I] = np.random.randn() * len(I)
    return (Y)


def gp_realization(T, err=0.0, tau=400.0,
                   basekernel=tinygp.kernels.quasisep.Exp,
                   seed=None):
    '''
    Generates a gaussian process at times T and errors err

    :param T:
    :param errmag:
    :param tau:
    :param basekernel:
    :param T_true:
    :param seed:

    Returns as lightcurve object
    '''
    if seed is None: seed = randint()

    # -----------------
    # Generate errors
    N = len(T)
    if isiter(err):
        E = err
    else:
        E = np.random.randn(N) * np.sqrt(err) + err
    E = abs(E)

    gp = GaussianProcess(basekernel(scale=tau), T)
    Y = gp.sample(jax.random.PRNGKey(seed))

    return (lightcurve(T=T, Y=Y, E=E))


# ================================================

class mock(object):
    '''
    Handy class for making mock data. When calling with _init_,
        defaultkwargs = {'tau':             400.0,
                         'cadence':         [7, 30],
                         'cadence_var':     [1, 5],
                         'season':          180,
                         'season_var':      14,
                         'N':               2048,
                         'maxtime':         360 * 5,
                         'lag':             30,
                         'E':               [0.01, 0.1],
                         'E_var':           [0.0, 0.0]
                         }
    '''

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

        self.seed = seed
        self.lc, self.lc_1, self.lc_2 = None, None, None
        self.lag = 0.0
        kwargs = defaultkwargs | kwargs
        self.args = {}

        for key in ['cadence', 'cadence_var', 'E', 'E_var']:
            if not (isiter(kwargs[key])): kwargs[key] = [kwargs[key], kwargs[key]]
        for key, var in zip(kwargs.keys(), kwargs.values()):
            self.__setattr__(key, var)
            self.args[key] = var

        self.generate(seed=seed)
        return

    def __call__(self, seed=0, **kwargs):
        self.generate(seed=seed)
        return (self.copy(seed))

    def generate_true(self, seed: int = 0):
        '''
        Generates an underlying true DRW signal and stores in the self attribute self.lc
        :param seed: seed for random generation
        :return: Array tuple (T,Y), underlying curve extending to maxtime + 2 * lag
        '''
        T = np.linspace(0.0, self.maxtime + self.lag * 2, self.N)
        Y = gp_realization(T, tau=self.tau, seed=seed).Y
        self.lc = lightcurve(T, Y)  # .trim(Tmin=0, Tmax=self.maxtime)
        return (T, Y)

    def generate(self, seed: int = 0):
        '''
        Generates a mock and sampled light-curve including a delayed response and stores in the self-attributes
        self.lc_1 and self.lc_2. Also returns as tuple (lc, lc_1, lc_2)
        :param seed: seed for random generation
        :return: lightcurve object
        '''

        T, Y = self.generate_true(seed=seed)

        T1 = mock_cadence(self.maxtime, seed, cadence=self.cadence[0], cadence_var=self.cadence_var[0],
                          season=self.season, season_var=self.season_var,
                          N=self.N)
        T2 = mock_cadence(self.maxtime, seed, cadence=self.cadence[1], cadence_var=self.cadence_var[1],
                          season=self.season, season_var=self.season_var,
                          N=self.N)

        Y1, Y2 = subsample(T, Y, T1), subsample(T + self.lag, Y, T2)
        E1, E2 = [np.random.randn(len(x)) * ev + e for x, ev, e in zip([T1, T2], self.E_var, self.E)]

        Y1 += np.random.randn(len(T1)) * abs(E1)
        Y2 += np.random.randn(len(T2)) * abs(E2)

        self.lc_1 = lightcurve(T1, Y1, E1)
        self.lc_2 = lightcurve(T2, Y2, E2)

        return (self.lc_1, self.lc_2)

    def copy(self, seed=None, **kwargs):
        '''
        Returns a copy of the mock while over-writing certain params.
        :param seed:
        :param kwargs:
        :return:
        '''
        if seed is None:
            seed = self.seed

        out = mock(seed=seed, **(self.args | kwargs))
        return (out)

    # ------------------------------
    # TEST UTILS
    def plot(self, axis=None, true_args={}, series_args={}, show=True):
        '''
        Plots the lightcurves and subsamples
        :param axis: matplotlib axis to plot to. If none will create new
        :param true_args: matplotlib plotting kwargs for the true underlying lightcurve
        :param series_args: matplotlib plotting kwargs for the observations
        :return: Plot axis
        '''

        # -----------------
        # Make / get axis
        if axis is None:
            f = plt.figure()
            axis = plt.gca()
            axis.grid()
            axis.set_xlabel("Time (days)")
            axis.set_ylabel("Signal Strength")

        # -----------------
        # Plot underlying curves
        true_args = {'lw': 0.5, 'c': ['tab:blue', 'tab:orange'], 'alpha': 0.3, 'label': ['True Signal', 'Response'],
                     } | true_args
        true_args_1 = true_args.copy()
        true_args_2 = true_args.copy()

        for key, val in zip(true_args.keys(), true_args.values()):
            if isiter(val) and len(val) > 1:
                true_args_1[key] = true_args[key][0]
                true_args_2[key] = true_args[key][1]
            else:
                true_args_1[key] = true_args[key]
                true_args_2[key] = true_args[key]

        lc_true_1, lc_true_2 = self.lc.delayed_copy(0, 0, self.maxtime), self.lc.delayed_copy(self.lag, 0, self.maxtime)
        # lc_true_2.T -= self.lag
        # lc_true_1.trim(0, self.maxtime), lc_true_2.trim(0, self.maxtime)

        axis.plot(lc_true_1.T, lc_true_1.Y, **true_args_1)
        axis.plot(lc_true_2.T, lc_true_2.Y, **true_args_2)

        # -----------------
        # Plot errorbars
        series_args = {'c': ['tab:blue', 'tab:orange'], 'alpha': 1.0, 'capsize': 2, 'lw': 1.5,
                       'label': ["Signal", "Response"]} | series_args
        series_args_1 = series_args.copy()
        series_args_2 = series_args.copy()

        for key, val in zip(series_args.keys(), series_args.values()):
            if isiter(val) and len(val) > 1:
                series_args_1[key] = series_args[key][0]
                series_args_2[key] = series_args[key][1]
            else:
                series_args_1[key] = series_args[key]
                series_args_2[key] = series_args[key]

        axis.errorbar(self.lc_1.T, self.lc_1.Y, self.lc_1.E, fmt='none',
                      **series_args_1
                      )
        axis.errorbar(self.lc_2.T, self.lc_2.Y, self.lc_2.E, fmt='none',
                      **series_args_2
                      )

        series_args_1.pop('capsize'), series_args_2.pop('capsize')
        axis.scatter(self.lc_1.T, self.lc_1.Y,
                     **(series_args_1 | {'s': 3, 'label': None})
                     )
        axis.scatter(self.lc_2.T, self.lc_2.Y,
                     **(series_args_2 | {'s': 3, 'label': None})
                     )

        if show: plt.show()
        return axis.get_figure()

    def corrected_plot(self, params={}, axis=None, true_args={}, series_args={}, show=False):
        params = self.params() | params
        corrected = self.copy()

        corrected.lc_2.T -= params['lag']
        corrected.lc_2 += params['rel_mean']
        corrected.lc_2 *= params['rel_amp']

        if 'alpha' in true_args.keys():
            if isiter(true_args['alpha']):
                true_args['alpha'][1] = 0.0
            else:
                true_args['alpha'] = [true_args['alpha'], 0.0]
        else:
            true_args |= {'alpha': [0.3, 0.0]}

        corrected.plot(axis=axis, true_args=true_args, series_args=series_args, show=show)

    def params(self):
        out = {
            'lag': self.lag,
            'logtau': np.log(self.tau),
            'logamp': 0.0,
            'rel_amp': 1.0,
            'mean': 0.0,
            'rel_mean': 0.0,
        }
        return (out)


# ================================================
# PRE-BAKED TEST CASES
# ================================================
# CASE A - WELL OBSERVED SMOOTH CURVES


def determ_gen(self, seed=0):
    f = lambda x: np.exp(-((x - 1000) / 2 / (64)) ** 2 / 2)
    X = np.linspace(0.0, self.maxtime + self.lag * 2, self.N)
    Y = f(X)
    self.lc = lightcurve(X, Y).trimmed_copy(Tmin=0, Tmax=self.maxtime)
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

        # ------------------
        plt.figure()
        x.corrected_plot(x.params(), axis=plt.gca(), true_args={'alpha': [0.15, 0.0]})
        plt.title("Corrected_plot for lag = %.2f" % x.lag)
        plt.grid()

        plt.show()
