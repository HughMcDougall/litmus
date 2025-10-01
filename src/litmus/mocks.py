"""
Some handy sets of mock data for use in testing

HM Apr 2024
"""

# ============================================
# IMPORTS

from copy import deepcopy as copy

import matplotlib

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from litmus.logging import logger

import jax

import tinygp
from tinygp import GaussianProcess

import litmus._types as _types
from litmus._utils import randint, isiter
from litmus.lightcurve import lightcurve


# ===================================================

def mock_cadence(maxtime, seed: int = 0, cadence: float = 7, cadence_var: float = 1, season: float = 180,
                 season_var: float = 14, N: int = 1024):
    """
    Returns time series X values for a mock signal
    :param maxtime: Length of simulation
    :param seed: Seed for randomization
    :param cadence: Average cadence of observations
    :param cadence_var: Standard deviation of the cadence
    :param season: Average length of the observation season (default 180 days)
    :param season_var: Standard deviation of the season length (default 14 days)
    :param N: Number of observations used prior to trimming. This is auto-tuned and is deprecated

    :return: returns as array of sample times
    """

    np.random.seed(seed)

    assert N > 0, "Invalid N. Must be <=0"

    # Generate Measurements
    while True:
        diffs = np.random.randn(N) * cadence_var / np.sqrt(2) + cadence
        T = np.cumsum(diffs)
        if T.max() <= maxtime:
            N *= 2
        else:
            break
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


def subsample(T: _types.ArrayN, Y: _types.ArrayN, Tsample: _types.ArrayN) -> _types.ArrayN:
    """
    Linearly interpolates between X and Y and returns interped Y's at positions Xsample

    :param T: Time values of time series to be interpolated
    :param Y: Y values of time series to be interpolated
    :param Tsample: Times to subample at
    :return: Elements of Y interpolated to times Tsample

    """
    out = np.interp(Tsample, T, Y)
    return (out)


def outly(Y, q) -> _types.ArrayN:
    """
    Returns a copy of Y with fraction 'q' elements replaced with
    unit - normally distributed outliers
    """
    I = np.random.rand(len(Y)) < q
    Y[I] = np.random.randn() * len(I)
    return (Y)


def gp_realization(T, err: _types.Union[float, _types.ArrayN] = 0.0, tau: float = 400.0,
                   basekernel: tinygp.kernels.quasisep = tinygp.kernels.quasisep.Exp,
                   seed=None) -> lightcurve:
    '''
    Generates a gaussian process at times T and errors err

    :param T: Time of observations
    :param err: Measurements uncertainty at observations. Must be float or array of same length as T
    :param tau: Timescale of the kernel
    :param basekernel: Kernel of the GP. Any tinyGP quasisep kernel
    :param seed:

    :return: Returns as lightcurve object
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

class mock(logger):
    """
    Handy class for making mock data. When calling with _init_, args can be passed as keyword arguments
    """

    def __init__(self, seed: int = 0, **kwargs):
        """
        :param seed: seed for randomization
        :param float tau: Timescale of the GP to be simulated
        :param float cadence: Mean cadence of the signals, either both or [signal 1, signal 2]. Defaults to [7 days, 30 days].
        :param float cadence_var: std of cadence of the signals, either both or [signal 1, signal 2]. Defaults to [1 day, 5 days].
        :param float season: Average season length. Defaults to 180 days.
        :param float season_var: std of season length. Defaults to 14 days.
        :param int N: Number of datapoints for the underlying realisation. Defaults to 2048.
        :param float maxtime: Max time of the underlying simulation. Defaults to 5 years.
        :param float lag: Lag for signal 2. Defaults to 30 days.
        :param float E: Mean measurement error for the signals, either both or [signal 1, signal 2]. Defaults to 1% and 10%.
        :param float E_var: Std of measurement error for the signals, either both or [signal 1, signal 2]. Defaults to 0%
        """
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

        logger.__init__(self)

        self.seed: int = seed
        self.lc, self.lc_1, self.lc_2 = None, None, None
        self.lag:float = 0.0
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

    def generate_true(self, seed: int = 0) -> (_types.ArrayN, _types.ArrayN):
        """
        Generates an underlying true DRW signal and stores in the self attribute self.lc
        :param seed: seed for random generation
        :return: Array tuple (T,Y), underlying curve extending to maxtime + 2 * lag
        """
        T = np.linspace(-self.lag*2, self.maxtime + self.lag * 2, self.N)
        Y = gp_realization(T, tau=self.tau, seed=seed).Y
        self.lc = lightcurve(T, Y)  # .trim(Tmin=0, Tmax=self.maxtime)
        return (T, Y)

    def generate(self, seed: int = 0) -> (lightcurve, lightcurve):
        """
        Generates a mock and sampled light-curve including a delayed response and stores in the self-attributes
        self.lc_1 and self.lc_2. Also returns as tuple (lc, lc_1, lc_2)
        :param seed: seed for random generation
        :return: lightcurve object
        """

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

    def copy(self, seed: int = None, **kwargs) -> _types.Self:
        """
        Returns a copy of the mock while over-writing certain params.
        :param seed: int seed for random generation
        :param kwargs: kwargs to pass to the new lightcurve object, will overwrite self.kwargs in the copy
        :return: A copy of self with kwargs and seed changed accordingly
        """
        if seed is None:
            seed = self.seed

        out = mock(seed=seed, **(self.args | kwargs))
        return (out)

    def swap_response(self, other: lightcurve) -> None:
        """
        Swaps the response lightcurves between this mock and its target.
        Over-writes target and self
        """

        self.lc_2, other.lc_2 = other.lc_2, self.lc_2
        self.lc, other.lc = None, None
        return

    # ------------------------------
    # TEST UTILS
    def plot(self, axis: matplotlib.axes.Axes = None, true_args: dict = {}, series_args: dict = {},
             show: bool = True) -> _types.Figure:
        """
        Plots the lightcurves and subsamples
        :param axis: matplotlib axis to plot to. If none will create new
        :param true_args: matplotlib plotting kwargs for the true underlying lightcurve
        :param series_args: matplotlib plotting kwargs for the observations
        :param show: If true will use plt.show() at the end fo the plot
        :return: Maplotlib figure
        """

        # -----------------
        # Make / get axis
        if axis is None:
            f = plt.figure()
            axis = plt.gca()
            axis.grid()
            axis.set_xlabel("Time (days)")
            axis.set_ylabel("Signal Strength")

        c0, c1, c2 = 'k', 'royalblue', 'orchid'
        # -----------------
        # Plot underlying curves
        true_args = {'lw': 0.5, 'c': [c1, c2], 'alpha': 0.3, 'label': ['True Signal', 'Response'],
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

        if self.lc is not None:
            lc_true_1, lc_true_2 = self.lc.delayed_copy(0, 0, self.maxtime), self.lc.delayed_copy(self.lag, 0,
                                                                                                  self.maxtime)

            axis.plot(lc_true_1.T, lc_true_1.Y, **true_args_1)
            axis.plot(lc_true_2.T, lc_true_2.Y, **true_args_2)

        # -----------------
        # Plot errorbars
        series_args = {'c': [c1, c2], 'alpha': 1.0, 'capsize': 2, 'lw': 1.5,
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

    def corrected_plot(self, params: dict = {}, axis: matplotlib.axis.Axis = None, true_args: dict = {},
                       series_args: dict = {}, show: bool = False) -> _types.Figure:
        """
        A copy of plot that offsets the displayed signals by self.lag to bring them into alignment.
        :param axis: matplotlib axis to plot to. If none will create new
        :param true_args: matplotlib plotting kwargs for the true underlying lightcurve
        :param series_args: matplotlib plotting kwargs for the observations
        :param show: If true will use plt.show() at the end fo the plot
        :return: matplotlib figure
        """
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
        """
        Helper utility that returns numpyro-like parameters.
        :return: Dict of param sites for gp_simple.
        """
        out = {
            'lag': self.lag,
            'logtau': np.log(self.tau),
            'logamp': 0.0,
            'rel_amp': 1.0,
            'mean': 0.0,
            'rel_mean': 0.0,
        }
        return (out)

    def lcs(self) -> (lightcurve, lightcurve):
        """
        Utility function for returning the two observed lightcurve, e.g. for fitter.fit(*mock.lcs())
        """
        return(self.lc_1, self.lc_2)


# ================================================
# PRE-BAKED TEST CASES
# ================================================
# CASE A - WELL OBSERVED SMOOTH CURVES


def _determ_gen(self, seed=0) -> (_types.ArrayN, _types.ArrayN):
    """
    Replaces the GP generation for the mock_A example to replace it with a nice gaussian curve
    """
    f = lambda x: np.exp(-((x - 1000) / 2 / (64)) ** 2 / 2)
    X = np.linspace(0.0, self.maxtime + self.lag * 2, self.N)
    Y = f(X)
    self.lc = lightcurve(X, Y).trimmed_copy(Tmin=0, Tmax=self.maxtime)
    return (X, Y)


# Change the way mock A generates a time series
mock_A = mock(season=None, lag=300)
"""Instead of a GP this mock has a clear bell-curve like hump. This works as a generous test-case of lag recovery methods"""
mock_A.generate_true = _types.MethodType(_determ_gen, mock_A)
mock_A()

# ================================================
# CASE B - SEASONAL GP
mock_B = mock(lag=256, maxtime=360 * 5, E=[0.01, 0.01], seed=2, season=180)
"""A standard oz-des like seasonal GP but with high SNR on the response LC. Good for testing failure states in the lag axis)"""

# ================================================
# CASE C - UN-SEASONAL GP
mock_C = mock(lag=128, maxtime=360 * 5, E=[0.01, 0.01], season=None)
"""A mock with no seasonal windowing"""