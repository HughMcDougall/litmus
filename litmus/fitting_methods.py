'''
Contains fitting procedures to be executed by the litmus class object

HM 24
'''

# ============================================
# IMPORTS
import sys
from functools import partial
import importlib.util
from warnings import warn

import numpy as np
from numpy import nan

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jaxopt
from jax.random import PRNGKey

# ------
# Samplers, stats etc

from tinygp import GaussianProcess

import numpyro
from numpyro import distributions as dist
from numpyro import infer

has_jaxns = importlib.util.find_spec('jaxns') is not None
has_polychord = importlib.util.find_spec('pypolychord') is not None
if has_jaxns:
    try:
        import tensorflow_probability.substrates.jax.distributions as tfpd
        import jaxns
    except:
        has_jaxns = False
        print("Warning! Something likely wrong in jaxns install or with numpyro integration", file=sys.stderr)
if has_polychord:
    import pypolychord

# ------
# Internal
import litmus._utils as _utils
import litmus.clustering as clustering
from litmus.ICCF_working import *

from litmus.models import quickprior
from litmus.models import _default_config
from litmus.models import stats_model
from litmus.lightcurve import lightcurve
from litmus.lin_scatter import linscatter


# ============================================

# Base fitting procedure class
class fitting_procedure(object):
    """
    Generic class for lag fitting procedures. Contains parent methods for setting properties
    """

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=True, **fit_params):

        if not hasattr(self, "_default_params"):
            self._default_params = {}

        self.stat_model = stat_model

        self.debug = debug
        self.verbose = verbose
        self.out_stream = out_stream
        self.err_stream = err_stream

        self.name = "Base Fitting Procedure"

        self.is_ready = False
        self.has_prefit = False
        self.has_run = False

        self.fitting_params = {} | self._default_params
        self.set_config(**(self._default_params | fit_params))

        self.seed = _utils.randint() if "seed" not in fit_params.keys() else fit_params['seed']
        self._tempseed = self.seed
        self._data = None

    # ----------------------
    '''
    Trying to set or get anything with a key in `fitting_params` or `results` will re-direct straight 
    to the corresponding dict entry.
    '''

    def __getattribute__(self, key):
        if key not in ["_default_params", "fitting_params"] \
                and hasattr(self, "_default_params") \
                and hasattr(self, "fitting_params") \
                and key in self._default_params.keys():
            return (self.fitting_params[key])
        else:
            return (super().__getattribute__(key))

    def __setattr__(self, key, value):
        if key not in ["_default_params", "fitting_params"] \
                and hasattr(self, "_default_params") \
                and hasattr(self, "fitting_params") \
                and key in self._default_params.keys():
            self.fitting_params[key] = value
            self.is_ready = False
            if self.has_run: self.msg_err(
                "Warning! Fitting parameter changed after a run. Can lead to unusual behaviour.")
        else:
            super().__setattr__(key, value)

    # ----------------------

    def reset(self):
        """
        Clears all memory and resets params to defaults
        """
        self.set_config(**self._default_params)

        self.has_run, self.is_ready = False, False

        return

    def set_config(self, **fit_params):
        """
        Configure fitting parameters for fitting_method() object
        Accepts any parameters present with a name in fitting_method.fitting_params
        Unlisted parameters will be ignored.
        """

        if self.debug: print("Doing config with keys", fit_params.keys())

        badkeys = [key for key in fit_params.keys() if key not in self._default_params.keys()]

        for key, val in zip(fit_params.keys(), fit_params.values()):
            if key in badkeys: continue

            # If something's changed, flag as having not run
            if self.has_run and val != self.__getattr__(key): self.has_run = False

            self.__setattr__(key, val)
            # self.fitting_params |= {key: val}
            if self.debug: print("\t set attr", key, file=self.out_stream)

        if len(badkeys) > 0:
            self.msg_err("Tried to configure bad keys:", *badkeys, delim='\t')
        return

    def readyup(self):
        """
        Performs pre-fit preparation calcs. Should only be called if not self.is_ready()
        """
        self.is_ready = True

    # ----------------------
    # Error message printing
    def msg_err(self, *x: str, end='\n', delim=' '):
        """
        Messages for when something has broken or been called incorrectly
        """
        if True:
            for a in x:
                print(a, file=self.err_stream, end=delim)

        print('', end=end, file=self.err_stream)
        return

    def msg_run(self, *x: str, end='\n', delim=' '):
        """
        Standard messages about when things are running
        """
        if self.verbose:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print('', end=end, file=self.out_stream)
        return

    def msg_debug(self, *x: str, end='\n', delim=' '):
        """
        Explicit messages to help debug when things are behaving strangely
        """
        if self.debug:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print('', end=end, file=self.out_stream, )
        return

    # ----------------------
    # Main methods

    def prefit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        """
        Fit lags
        :param lc_1: Lightcurve 1 (Main)
        :param lc_2: Lightcurve 2 (Response)
        :param seed: A random seed for feeding to the fitting process. If none, will select randomly
        """

        self.has_prefit = True

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        """
        Fit lags
        :param lc_1: Lightcurve 1 (Main)
        :param lc_2: Lightcurve 2 (Response)
        :param seed: A random seed for feeding to the fitting process. If none, will select randomly
        """

        # Sanity checks inherited by all subclasses
        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
            self._tempseed = _utils.randint()
        seed = self._tempseed

        self._data = self.stat_model.lc_to_data(lc_1, lc_2)
        data = self._data

        # An error message raised if this fitting procedure doesn't have .fit()
        if self.__class__.fit == fitting_procedure.fit:
            self.msg_err("Fitting \"%s\" method does not have method .fit() implemented" % self.name)

        return

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        '''
        Returns MCMC-like posterior samples
        :param N: Number of samples to return. If None, return all
        :param seed: Random seed for any stochastic elements
        :param importance_sampling: If true, will weight the results by
        :return:
        '''

        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.fit == fitting_procedure.fit:
            self.msg_err("Fitting \"%s\" method does not have method .get_samples() implemented" % self.name)

    def get_evidence(self, seed: int = None) -> [float, float, float]:
        """
        Returns the estimated evidence for the fit model. Returns as array-like [Z,dZ-,dZ+]
        """

        if not self.is_ready: self.readyup()
        if not self.has_run: self.msg_err("Warning! Tried to call get_evidence without running first!")

        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_evidence == fitting_procedure.get_evidence:
            self.msg_err("Fitting \"%s\" method does not have method .get_evidence() implemented" % self.name)

        return (np.array([0.0, 0.0, 0.0]))

    def get_information(self, seed: int = None) -> [float, float, float]:
        """
        Returns an estimate of the information (KL divergence relative to prior). Returns as array-like [I,dI-,dI+]
        """

        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_information == fitting_procedure.get_information:
            self.msg_err("Fitting \"%s\" method does not have method .get_information() implemented" % self.name)

        return (np.array([0.0, 0.0, 0.0]))

    def get_peaks(self, seed=None):
        """
        Returns the maximum posterior position in parameter space
        """

        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_peaks == fitting_procedure.get_peaks:
            self.msg_err("Fitting \"%s\" method does not have method .get_peaks() implemented" % self.name)

        return ({}, np.array([]))


# ============================================
# ICCF fitting procedure

class ICCF(fitting_procedure):
    """
    Fit lags using interpolated cross correlation function
    todo
        - Add p value, false positive and evidence estimates
    """

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {
                'Nboot': 512,
                'Nterp': 2014,
                'Nlags': 512,
            }

        super().__init__(**args_in)

        self.name = "ICCF Fitting Procedure"
        self.lags = np.zeros(self.Nterp)

        # -----------------------------------
        self.samples = np.zeros(self.Nboot)
        self.correl_curve = np.zeros(self.Nterp)
        self.lag_mean = 0.0
        self.lag_err = 0.0

    # -------------------------
    def set_config(self, **fit_params):
        super().set_config(**fit_params)

    def readyup(self):
        super().readyup()
        '''
        # self.lags = jnp.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags)
        self.lags = np.random.randn(self.Nlags) * self.stat_model.prior_ranges['lag'].ptp() + \
                    self.stat_model.prior_ranges['lag'][0]
        '''
        self.is_ready = True
        self.has_prefit = False

    # -------------------------
    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        # Unpack lightcurve
        X1, Y1, E1 = lc_1.T, lc_1.Y, lc_1.E
        X2, Y2, E2 = lc_2.T, lc_2.Y, lc_2.E

        # Get interpolated correlation for all un-bootstrapped data
        self.correls = correlfunc_jax_vmapped(self.lags, X1, Y1, X2, Y2, self.Nterp)

        # Do bootstrap fitting
        lagrange = jnp.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags)
        jax_samples = correl_func_boot_jax_wrapper_nomap(lagrange, X1, Y1, X2, Y2, E1, E2,
                                                         Nterp=self.Nterp,
                                                         Nboot=self.Nboot)

        # Store Results
        self.samples = jax_samples
        self.lag_mean, self.lag_err = jax_samples.mean(), jax_samples.std()

        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed

        if importance_sampling:
            self.msg_err("Warning! Cannot use important sampling with ICCF. Try implementing manually")
            return
        # -------------------

        # Return entire sample chain or sub-set of samples
        if N is None:
            return ({'lag': self.samples})
        else:
            if N > self.Nboot:
                self.msg_err(
                    "Warning, tried to get %i sub-samples from %i boot-strap iterations in ICCF" % (N, self.Nboot),
                )
            return ({'lag': np.random.choice(a=self.samples, size=N, replace=True)})

    def get_peaks(self, seed: int = None) -> ({float: [float]}, [float]):
        # -------------------
        fitting_procedure.get_peaks(**locals())
        seed = self._tempseed
        # --------------
        out = self.lags[np.argmax(self.correls)]
        return ({'lag': np.array([out])})


# ============================================
# Random Prior Sampling

class prior_sampling(fitting_procedure):
    """
    Randomly samples from the prior and weights with importance sampling.
    The crudest available sampler. For test purposes only.
    """

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        # ------------------------------------
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {
                'Nsamples': 4096
            }

        super().__init__(**args_in)
        # ------------------------------------

        self.name = "Prior Sampling Fitting Procedure"

        self.samples = np.zeros(self.Nsamples)
        self.log_likes = np.zeros(self.Nsamples)
        self.weights = np.zeros(self.Nsamples)

    # --------------
    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        # Generate samples & calculate likelihoods todo - These currently give posterior densities and not log likes
        data = self.stat_model.lc_to_data(lc_1, lc_2)
        samples = self.stat_model.prior_sample(num_samples=self.Nsamples, seed=seed)
        log_density = self.stat_model.log_density(data=data, params=samples)
        log_prior = self.stat_model.log_prior(params=samples)
        log_likes = log_density - log_prior
        likes = np.exp(log_likes)

        # Store results

        self.log_prior = log_prior
        self.log_likes = log_likes
        self.log_density = log_density
        self.samples = samples
        self.weights = likes / likes.sum()

        # Mark as having completed a run
        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = True) -> {str: [float]}:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------

        if N is None:
            N = self.Nsamples
        else:
            if N > self.Nsamples:
                self.msg_err("Warning, tried to get %i sub-samples from %i samples" % (N, self.Nsamples))

        if importance_sampling:
            weights = self.weights / self.weights.sum()
        else:
            weights = None

        I = np.random.choice(a=np.arange(self.Nsamples), size=N, replace=True,
                             p=weights)
        return ({
            key: val[I] for key, val in zip(self.samples.keys(), self.samples.values())
        })

    def get_evidence(self, seed=None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------
        density = np.exp(self.log_density - self.log_density.max())

        Z = density.mean() * self.stat_model.prior_volume * np.exp(self.log_density.max())
        uncert = density.std() / np.sqrt(self.Nsamples) * self.stat_model.prior_volume

        return (np.array([Z, -uncert, uncert]))

    def get_information(self, seed: int = None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------
        info_partial = np.random.choice(self.log_density - self.log_prior, self.Nsamples,
                                        p=self.weights)
        info = info_partial.mean() * self.stat_model.prior_volume
        uncert = info_partial.std() / np.sqrt(self.Nsamples) * self.stat_model.prior_volume

        return (np.array([info, -uncert, uncert]))


# ============================================
# Nested Sampling

class nested_sampling(fitting_procedure):
    """
    Access to nested sampling. NOT FULLY IMPLEMENTED
    In version 1.0.0 this will use either jaxns or pypolychord to perform direct nested sampling.
    """

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {
                'num_live_points': 500,
                'max_samples': 10_000,
                'num_parallel_samplers': 1,
                'evidence_uncert': 1E-3,
                'live_evidence_frac': np.log(1 + 1e-3),
            }

        super().__init__(**args_in)

        self.name = "Nested Sampling Fitting Procedure"

        self.sampler = None

        self._jaxnsmodel = None
        self._jaxnsresults = None
        self._jaxnstermination = None
        self._jaxnsresults = None

        self.logevidence = jnp.zeros(3)
        self.priorvolume = 0.0

    def prefit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # ---------------------
        fitting_procedure.prefit(**locals())
        if seed is None: seed = _utils.randint()
        # ---------------------

        # Get uniform prior bounds
        bounds = np.array([self.stat_model.prior_ranges[key] for key in self.stat_model.paramnames()])
        lo, hi = jnp.array(bounds[:, 0]), jnp.array(bounds[:, 1])

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        # Construct jaxns friendly prior & likelihood
        def prior_model():
            x = yield jaxns.Prior(tfpd.Uniform(low=lo, high=hi), name='x')
            return x

        def log_likelihood(x):
            params = _utils.dict_unpack(x, keys=self.stat_model.paramnames())
            with numpyro.handlers.block(hide=self.stat_model.paramnames()):
                LL = self.stat_model._log_likelihood(params, data=data)
            return LL

        # jaxns object setup
        self._jaxnsmodel = jaxns.Model(prior_model=prior_model,
                                       log_likelihood=log_likelihood,
                                       )

        self._jaxnstermination = jaxns.TerminationCondition(
            dlogZ=self.evidence_uncert,
            max_samples=self.max_samples,
        )

        # Build jaxns Nested Sampler
        self.sampler = jaxns.NestedSampler(
            model=self._jaxnsmodel,
            max_samples=self.max_samples,
            verbose=self.debug,
            num_live_points=self.num_live_points,
            num_parallel_workers=self.num_parallel_samplers,
            difficult_model=True,
        )

        self.has_prefit = True

    # --------------
    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # ---------------------
        fitting_procedure.fit(**locals())
        if seed is None: seed = _utils.randint()
        # ---------------------
        if not self.has_prefit:
            self.prefit(lc_1, lc_2, seed)
        self.readyup()
        # ---------------------

        # -----------------
        # Run the sampler!
        termination_reason, state = self.sampler(jax.random.PRNGKey(seed),
                                                 self._jaxnstermination)

        # -----------------
        # Extract & save results
        self._jaxnsresults = self.sampler.to_results(
            termination_reason=termination_reason,
            state=state
        )

        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        if seed is None: seed = _utils.randint()

        samples, logweights = self._jaxnsresults.samples['x'], self._jaxnsresults.log_dp_mean

        if N is None:
            if importance_sampling:
                out = {key: samples.T[i] for i, key in enumerate(self.stat_model.paramnames())}
                return out
            else:
                N = samples.shape[0]

        if importance_sampling:
            logweights = jnp.zeros_like(logweights)

        samples = jaxns.resample(
            key=jax.random.PRNGKey(seed),
            samples=samples,
            log_weights=logweights,
            S=N
        )

        out = {key: samples.T[i] for i, key in enumerate(self.stat_model.paramnames())}

        return (out)

    def get_evidence(self, seed: int = None) -> [float, float, float]:
        '''
        Returns the -1, 0 and +1 sigma values for model evidence from nested sampling.
        This represents an estimate of numerical uncertainty
        '''

        if seed is None: seed = _utils.randint()

        l, l_e = self._jaxnsresults.log_Z_mean, self._jaxnsresults.log_Z_uncert

        out = np.exp([
            l,
            l - l_e,
            l + l_e
        ])

        out -= np.array([0, out[0], out[0]])

        return (out)

    def get_information(self, seed: int = None) -> [float, float, float]:
        '''
        Use the Nested Sampling shells to estimate the model information relative to prior
        '''
        # todo - this is outmoded

        if seed is None: seed = _utils.randint()

        NS = self.sampler
        samples, logweights = self._jaxnsresults.samples, self._jaxnsresults.log_dp_mean

        weights = np.exp(logweights)
        weights /= weights.sum()

        log_density = self._jaxnsresults.log_posterior_density
        prior_values = self.stat_model.log_prior(samples)

        info = np.sum((log_density - prior_values) * weights)

        partial_info = np.random.choice((log_density - prior_values), len(log_density), p=weights)
        uncert = partial_info.std() / np.sqrt(len(log_density))

        return (np.array(info, uncert, uncert))

    def get_peaks(self, seed: int = None) -> ({str: [float]}, float):

        # todo - this is outmoded

        # ---------------------
        if seed is None: seed = _utils.randint()
        # ---------------------

        self.msg_err("get_peaks currently placeholder.")
        return ({key: np.array([]) for key in self.stat_model.paramnames()}, np.array([]))

        # ---------------------

        NS = self.sampler
        samples = self.get_samples()
        log_densities = NS._results.log_posterior_density

        # Find clusters
        indices = clustering.clusterfind_1D(samples['lag'])

        # Break samples and log-densities up into clusters
        sorted_samples = clustering.sort_by_cluster(samples, indices)
        sort_logdens = clustering.sort_by_cluster(log_densities, indices)

        Nclusters = len(sorted_samples)

        # Make an empty dictionary to store positions in
        peak_locations = {key: np.zeros([Nclusters]) for key in samples.keys()}
        peaklikes = np.zeros([Nclusters])

        for i, group, lds in enumerate(sorted_samples, sort_logdens):
            j = np.argmax(lds)
            for key in samples.keys():
                peak_locations[key][i] = group[key][j]
            peaklikes[i] = lds[j]

        return (peak_locations, peaklikes)

    def diagnostics(self):
        jaxns.plot_diagnostics(self._jaxnsresults)
        return


# ------------------------------------------------------
class hessian_scan(fitting_procedure):
    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {
                'Nlags': 1024,
                'opt_tol': 1E-3,
                'opt_tol_init': 1E-5,
                'step_size': 0.001,
                'constrained_domain': False,
                'max_opt_eval': 1_000,
                'max_opt_eval_init': 5_000,
                'LL_threshold': 100.0,
                'init_samples': 5_000,
                'grid_bunching': 0.5,
                'grid_relaxation': 0.5,
                'grid_depth': None,
                'grid_Nterp': None,
                'reverse': True,
                'optimizer_args_init': {},
                'optimizer_args': {},
                'seed_params': {},
                'precondition': 'diag'
            }

        super().__init__(**args_in)

        # -----------------------------------

        self.name = "Hessian Scan Fitting Procedure"

        self.scan_peaks = None
        self.evidence = None

        self.lags = np.zeros(self.Nlags)
        self.converged = np.zeros_like(self.lags, dtype=bool)

        self.diagnostic_hessians = None
        self.diagnostic_grads = None
        self.diagnostic_tols = None

        self.solver = None

        self.params_toscan = self.stat_model.free_params()
        self.params_toscan.remove('lag')

        self.estmap_params = {}

    def readyup(self):

        # Get grid properties
        if self.grid_depth is None:
            self.grid_depth = int(1 / (1 - self.grid_relaxation) * 5)
        if self.grid_Nterp is None:
            self.grid_Nterp = self.Nlags * 10

        # Make list of lags for scanning
        self.lags = np.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags + 1, endpoint=False)[1:]
        self.converged = np.zeros_like(self.lags, dtype=bool)

        free_dims = len(self.stat_model.free_params())
        self.scan_peaks = {key: np.array([]) for key in self.stat_model.paramnames()}
        self.diagnostic_hessians = []
        self.diagnostic_grads = []
        self.diagnostic_tols = []

        self.params_toscan = [key for key in self.stat_model.paramnames() if
                              key not in ['lag'] and key in self.stat_model.free_params()
                              ]

        self.is_ready = True

    # --------------

    def estimate_MAP(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        '''
        :param lc_1:
        :param lc_2:
        :param seed:
        :return:
        '''
        # todo - reformat this and the make_grid for better consistency in drawing from estmap and seed params

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        # ----------------------------------
        # Find seed for optimization if not supplies
        if self.stat_model.free_params() != self.seed_params.keys():
            seed_params, ll_start = self.stat_model.find_seed(data, guesses=self.init_samples, fixed=self.seed_params)

            self.msg_run("Beginning scan at constrained-space position:")
            for it in seed_params.items():
                self.msg_run('\t %s: \t %.2f' % (it[0], it[1]))
            self.msg_run(
                "Log-Density for this is: %.2f" % ll_start)

        else:
            seed_params = self.seed_params
            ll_start = self.stat_model.log_density(seed_params,
                                                   data=data
                                                   )

        # ----------------------------------
        # SCANNING FOR OPT

        self.msg_run("Moving to new location...")
        estmap_params = self.stat_model.scan(start_params=seed_params,
                                             optim_params=self.stat_model.free_params(),
                                             data=data,
                                             optim_kwargs=self.optimizer_args_init,
                                             precondition=self.precondition
                                             )
        ll_end = self.stat_model.log_density(estmap_params,
                                             data=data
                                             )

        # ----------------------------------
        # CHECKING OUTPUTS

        if ll_end < ll_start:
            self.msg_err("Warning! Optimization seems to have diverged. Defaulting to seed params. \n"
                         "Please consider running with different optim_init inputs")
            estmap_params = seed_params

        # ----------------------------------
        # CHECKING OUTPUTS

        self.msg_run("Optimizer settled at new fit:")
        for it in estmap_params.items():
            self.msg_run('\t %s: \t %.2f' % (it[0], it[1]))

        self.msg_run(
            "Log-Density for this is: %.2f" % ll_end
        )

        return estmap_params

    def make_grid(self, data, seed_params=None):
        '''
        :param data:
        :param seed_params:
        :return:
        '''
        # todo - reformat this and the make_grid for better consistency in drawing from estmap and seed params

        if not self.is_ready: self.readyup()

        # If no seed parameters specified, use stored
        if seed_params is None:
            seed_params = self.estmap_params

        # If these params are incomplete, use find_seed to complete them
        if seed_params.keys() != self.stat_model.paramnames():
            seed_params, llstart = self.stat_model.find_seed(data, guesses=self.init_samples, fixed=seed_params)

        lags = np.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags + 1, endpoint=False)[1:]
        lag_terp = np.linspace(*self.stat_model.prior_ranges['lag'], self.grid_Nterp)

        percentiles_old = np.linspace(0, 1, self.grid_Nterp)
        for i in range(self.grid_depth):
            params = _utils.dict_extend(seed_params, {'lag': lags})
            density = np.exp(self.stat_model.log_density(params, data))
            density /= density.sum()

            density_terp = np.interp(lag_terp, lags, density, left=0, right=0)
            gets = np.linspace(0, 1, self.grid_Nterp)

            percentiles_new = np.cumsum(density_terp) * self.grid_bunching + gets * (1 - self.grid_bunching)
            percentiles = percentiles_old * self.grid_relaxation + percentiles_new * (1 - self.grid_relaxation)
            percentiles /= percentiles.max()
            percentiles_old = percentiles.copy()

            lags = np.interp(np.linspace(0, 1, self.Nlags), percentiles, lag_terp, left=0, right=lag_terp.max())

        return (lags)

    # --------------

    def prefit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.prefit(**locals())
        seed = self._tempseed
        # -------------------

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        # ----------------------------------
        # Estimate the MAP

        self.estmap_params = self.estimate_MAP(lc_1, lc_2, seed)
        self.estmap_tol = self.stat_model.opt_tol(self.estmap_params, data,
                                                  integrate_axes=self.stat_model.free_params())
        self.msg_run("Estimated to be within ±%.2eσ of local optimum" % self.estmap_tol)
        # ----------------------------------

        # Make a grid

        lags = self.make_grid(data, seed_params=self.estmap_params)
        self.lags = lags

        self.has_prefit = True

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        self.msg_run("Starting Hessian Scan")

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        if not self.has_prefit:
            self.prefit(lc_1, lc_2, seed)
        # ----------------------------------
        # Run over 'self.lags' and scan all positions
        scanned_optima = []

        best_params = self.estmap_params.copy()
        params_toscan = self.params_toscan

        solver, runsolver, [converter, deconverter, optfunc, runsolver_jit] = self.stat_model._scanner(data,
                                                                                                       optim_params=params_toscan,
                                                                                                       optim_kwargs=self.optimizer_args,
                                                                                                       return_aux=True
                                                                                                       )
        x0, y0 = converter(best_params)
        state = solver.init_state(x0, y0, data)
        self.solver = solver

        lags_forscan = self.lags
        if self.reverse: lags_forscan = lags_forscan[::-1]

        tols = []
        for i, lag in enumerate(lags_forscan):
            self.msg_run(":" * 23)
            self.msg_run("Scanning at lag=%.2f ..." % lag)

            # Get current param site in packed-function friendly terms
            opt_params, aux_data, state = runsolver_jit(solver, best_params | {'lag': lag}, state)

            # --------------
            # Check if the optimization has succeeded or broken

            l_1 = self.stat_model.log_density(best_params | {'lag': lag}, data)
            l_2 = self.stat_model.log_density(opt_params | {'lag': lag}, data)
            diverged = np.any(np.isinf(np.array([x for x in self.stat_model.to_uncon(opt_params).values()])))

            self.msg_run("Change of %.2f against %.2f" % (l_2 - l_1, self.LL_threshold))

            if l_2 - l_1 > -self.LL_threshold and not diverged:
                self.msg_run("Seems to have converged at iteration %i / %i" % (i, self.Nlags))
                best_params = opt_params

                self.converged[i] = True

                scanned_optima.append(opt_params.copy())

                self.diagnostic_grads.append(aux_data['grad'])
                uncon_params = self.stat_model.to_uncon(opt_params)
                H = self.stat_model.log_density_uncon_hess(uncon_params, data, keys=params_toscan)
                self.diagnostic_hessians.append(H)
                try:
                    tol = np.dot(aux_data['grad'],
                                 np.dot(np.linalg.inv(H), aux_data['grad'])
                                 )
                    tols.append(tol)
                except:
                    self.converged[i] = False
                    self.msg_run("Seems to have severely diverged at iteration %i / %i" % (i, self.Nlags))

            else:
                self.msg_run("Unable to converge at iteration %i / %i" % (i, self.Nlags))

        self.msg_run("Scanning Complete. Calculating laplace integrals...")
        self.diagnostic_tols = np.sqrt(abs(np.array(tols)))

        self.scan_peaks = _utils.dict_combine(scanned_optima)

        # For each of these peaks, estimate the evidence
        # todo - add a max LL significance to cut down on evals
        # todo - vmap and parallelize
        Zs = []
        integrate_axes = self.stat_model.free_params().copy()

        if 'lag' in integrate_axes: integrate_axes.remove('lag')
        for params in scanned_optima:

            # todo - this is partially redundant as we already have the hessians from above

            Z_lap = self.stat_model.laplace_log_evidence(params=params,
                                                         data=data,
                                                         integrate_axes=integrate_axes,
                                                         constrained=self.constrained_domain
                                                         )
            if not self.constrained_domain: Z_lap += self.stat_model.uncon_grad(params)
            Zs.append(Z_lap)

        self.log_evidences = np.array(Zs)

        self.has_run = True

        self.msg_run("Hessian Scan Fitting complete.", "-" * 23, "-" * 23, delim='\n')

    def refit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        peaks = _utils.dict_divide(self.scan_peaks)
        I = np.arange(len(peaks))
        select = np.argwhere(self.diagnostic_tols > self.opt_tol).squeeze()
        if not (_utils.isiter(select)): select = np.array([select])

        peaks, I = np.array(peaks)[select], I[select]

        self.msg_run("Doing re-fitting of %i lags" % len(peaks))
        newtols = []
        for j, i, peak in zip(range(len(I)), I, peaks):

            self.msg_run(":" * 23, "Refitting lag %i/%i at lag %.2f" % (j, len(peaks), peak['lag']), delim='\n')
            old_tol = self.diagnostic_tols[i]

            new_peak = self.stat_model.scan(start_params=peak,
                                            data=data,
                                            optim_kwargs=self.optimizer_args
                                            )
            new_peak_uncon = self.stat_model.to_uncon(new_peak)
            new_grad = self.stat_model.log_density_uncon_grad(new_peak_uncon, data)
            new_hessian = self.stat_model.log_density_uncon_hess(new_peak_uncon, data)

            J = np.where([key in self.params_toscan for key in self.stat_model.paramnames()])[0]
            new_grad = _utils.dict_pack(new_grad, keys=self.params_toscan)
            if len(J) > 1:
                new_hessian = new_hessian[J, :][:, J]
            elif len(J) == 1:
                new_hessian = new_hessian[J, :][:, J]
            try:
                Hinv = np.linalg.inv(new_hessian)
            except:
                self.msg_run("Optimization failed on %i/%i" % (j, len(peaks)))
                continue

            tol = np.dot(new_grad, np.dot(Hinv, new_grad))
            tol = np.sqrt(abs(tol))
            if tol < old_tol:
                self.diagnostic_grads[i] = new_grad
                self.diagnostic_hessians[i] = new_hessian
                self.diagnostic_tols[i] = tol
                self.msg_run("Settled at new tol %.2e" % tol)
            else:
                self.msg_run(
                    "Something went wrong at this refit! Consider changing the optimizer_args and trying again")
        self.msg_run("Refitting complete.")

    def diagnostics(self, plot=True):
        '''
        Runs some diagnostics for convergence
        :return:
        '''

        loss = self.diagnostic_tols

        lagplot = self.scan_peaks['lag']

        # ---------
        fig = plt.figure()
        plt.ylabel("Loss Norm, $ \\vert \Delta x / \sigma_x \\vert$")
        plt.xlabel("Scan Lag No.")
        plt.plot(lagplot, loss, 'o-', c='k', label="Scan Losses")
        plt.scatter(self.estmap_params['lag'], self.estmap_tol, c='r', marker='x', s=40, label="Initial MAP Scan Loss")
        plt.axhline(self.opt_tol, ls='--', c='k', label="Nominal Tolerance Limit")
        plt.legend(loc='best')

        fig.text(.5, .05, "How far each optimization slice is from its peak. Lower is good.", ha='center')
        plt.yscale('log')
        plt.grid()
        plt.show()

    def get_evidence(self, seed: int = None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_evidence(**locals())
        seed = self._tempseed
        # -------------------

        if not isinstance(self, SVI_scan):
            select = np.argwhere(self.diagnostic_tols < self.opt_tol).squeeze()
        else:
            select = np.ones_like(self.log_evidences, dtype=bool)
        if not (_utils.isiter(select)): select = np.array([select])

        # Calculating Evidence
        lags_forint = self.scan_peaks['lag'][select]
        minlag, maxlag = self.stat_model.prior_ranges['lag']
        if self.reverse:
            minlag, maxlag = maxlag, minlag
        dlag = [*np.diff(lags_forint) / 2, 0]
        dlag[1:] += np.diff(lags_forint) / 2
        dlag[0] += lags_forint.min() - minlag
        dlag[-1] += maxlag - lags_forint.max()

        if sum(dlag) == 0: dlag = 1.0

        dZ = np.exp(self.log_evidences[select] - self.log_evidences[select].max())
        Z = (dZ * dlag).sum()
        Z *= np.exp(self.log_evidences[select].max())

        # -------------------------------------
        # Get Uncertainties

        # Estimate uncertainty from ~dt^2 error scaling
        Z_subsample = np.trapz(dZ[::2], lags_forint[::2])
        Z_subsample *= np.exp(self.log_evidences[select].max())
        uncert_numeric = abs(Z - Z_subsample) / 3

        if not isinstance(self, SVI_scan):
            uncert_tol = (dZ * self.diagnostic_tols[select]).sum()
            uncert_tol *= np.exp(self.log_evidences[select].max())

            self.msg_debug("Evidence Est: \t %.2e" % Z)
            self.msg_debug(
                "Evidence uncerts: \n Numeric: \t %.2e \n Convergence: \t %.2e" % (uncert_numeric, uncert_tol))

            uncert_plus = np.sqrt(np.square(np.array([uncert_numeric, uncert_tol])).sum())
            uncert_minus = uncert_numeric
        else:
            uncert_plus, uncert_minus = uncert_numeric, uncert_numeric

        return (np.array([Z, uncert_minus, uncert_plus]))

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------

        # Get weights and peaks etc
        Npeaks = len(self.log_evidences)

        lags_forint = self.scan_peaks['lag']
        minlag, maxlag = self.stat_model.prior_ranges['lag']
        if self.reverse:
            minlag, maxlag = maxlag, minlag

        Y = np.exp(self.log_evidences - self.log_evidences.max())

        dlag = [*np.diff(lags_forint) / 2, 0]
        dlag[1:] += np.diff(lags_forint) / 2
        dlag[0] += lags_forint.min() - minlag
        dlag[-1] += maxlag - lags_forint.max()

        if sum(dlag) == 0:
            dlag = 1.0

        weights = Y * dlag
        weights /= weights.sum()

        # Get hessians and peak locations
        covars = np.array([np.linalg.inv(H) for H in self.diagnostic_hessians])
        peaks = _utils.dict_divide(self.stat_model.to_uncon(self.scan_peaks))

        # Get hessians and peak locations
        I = np.random.choice(range(Npeaks), N, replace=True, p=weights)

        to_choose = [(I == i).sum() for i in range(Npeaks)]  # number of samples to draw from peak i

        # Sweep over scan peaks and add scatter
        outs = []
        for i in range(Npeaks):
            if to_choose[i] != 0:
                # Get normal dist properties in uncon space in vector form
                mu = _utils.dict_pack(peaks[i], keys=self.params_toscan)
                cov = covars[i]

                # Generate samples
                samps = np.random.multivariate_normal(mean=mu, cov=cov, size=to_choose[i])
                samps = _utils.dict_unpack(samps.T, keys=self.params_toscan, recursive=False)
                samps = _utils.dict_extend(peaks[i], samps)

                # Reconvert to constrained space
                samps = self.stat_model.to_con(samps)

                # -------------
                # Add linear interpolation 'smudging' to lags
                if Npeaks > 1:

                    # Get nodes
                    tnow, ynow = lags_forint[i], Y[i]
                    if i == 0:
                        yprev, ynext = ynow, Y[i + 1]
                        tprev, tnext = min(self.stat_model.prior_ranges['lag']), lags_forint[i + 1]
                    elif i == Npeaks - 1:
                        yprev, ynext = Y[i - 1], ynow
                        tprev, tnext = lags_forint[i - 1], max(self.stat_model.prior_ranges['lag'])
                    else:
                        yprev, ynext = Y[i - 1], Y[i + 1]
                        tprev, tnext = lags_forint[i - 1], lags_forint[i + 1]
                    # --

                    # Perform CDF shift
                    Ti, Yi = [tprev, tnow, tnext], [yprev, ynow, ynext]
                    tshift = linscatter(Ti, Yi, N=to_choose[i])

                    if np.isnan(tshift).any():
                        self.msg_err("Something wrong with the lag shift at node %i in sample generation" % i)
                    else:
                        samps['lag'] += tshift
                # -------------

                if np.isnan(samps['lag']).any():
                    self.msg_err("Something wrong with the lags at node %i in sample generation" % i)
                else:
                    outs.append(samps)

        outs = {key: np.concatenate([out[key] for out in outs]) for key in self.stat_model.paramnames()}
        return (outs)


# -----------------------------------
class SVI_scan(hessian_scan):
    '''
    An alternative to hessian_scan that fits each slice with stochastic variational
    inference instead of the laplace approximation. May be slower.
    '''

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {
                'Nlags': 1024,
                'opt_tol': 1E-3,
                'opt_tol_init': 1E-5,
                'step_size': 0.001,
                'constrained_domain': False,
                'max_opt_eval': 1_000,
                'max_opt_eval_init': 5_000,
                'ELBO_threshold': 100.0,
                'init_samples': 5_000,
                'grid_bunching': 0.5,
                'grid_relaxation': 0.5,
                'grid_depth': None,
                'grid_Nterp': None,
                'reverse': False,
                'optimizer_args_init': {},
                'seed_params': {},
                'ELBO_optimstep': 5E-3,
                'ELBO_particles': 128,
                'ELBO_Nsteps': 100,
                'ELBO_Nsteps_init': 1_000,
                'ELBO_fraction': 0.1,
                'precondition': 'half-eig'
            }

        super().__init__(**args_in)

        # -----------------------------

        self.name = "SVI Scan Fitting Procedure"

        self.ELBOS = []
        self.diagnostic_losses = []
        self.diagnostic_loss_init = []

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        self.msg_run("Starting SVI Scan")

        if not self.has_prefit:
            self.prefit(lc_1, lc_2, seed)

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        # ----------------------------------
        # Estimate the MAP and its hessian

        estmap_uncon = self.stat_model.to_uncon(self.estmap_params)

        fix_param_dict_con = {key: self.estmap_params[key] for key in self.stat_model.fixed_params()}
        fix_param_dict_uncon = {key: estmap_uncon[key] for key in self.stat_model.fixed_params()}

        init_hess = -1 * self.stat_model.log_density_uncon_hess(estmap_uncon, data=data, keys=self.params_toscan)

        # ----------------------------------
        # Convert these into SVI friendly objects and fit an SVI at the map
        self.msg_run("Performing SVI slice at the MAP estimate")
        init_loc = _utils.dict_pack(estmap_uncon, keys=self.params_toscan)
        init_tril = jnp.linalg.cholesky(jnp.linalg.inv(init_hess))

        self.msg_debug("\t Constructing slice model")

        def slice_function(data, lag):

            params = {}
            for key in self.stat_model.free_params():
                if key != 'lag':
                    val = quickprior(self.stat_model, key)
                    params |= {key: val}
            params |= {'lag': lag}
            params |= fix_param_dict_con

            with numpyro.handlers.block(hide=self.stat_model.paramnames()):
                LL = self.stat_model._log_likelihood(params, data)
            numpyro.factor('log_likelihood', LL)

        # SVI settup
        self.msg_debug("\t Constructing and running optimizer and SVI guides")
        optimizer = numpyro.optim.Adam(step_size=self.ELBO_optimstep)
        autoguide = numpyro.infer.autoguide.AutoMultivariateNormal(slice_function)
        autosvi = numpyro.infer.SVI(slice_function, autoguide, optim=optimizer,
                                    loss=numpyro.infer.Trace_ELBO(self.ELBO_particles),
                                    )

        self.msg_debug("\t Running SVI")
        MAP_SVI_results = autosvi.run(jax.random.PRNGKey(seed), self.ELBO_Nsteps_init,
                                      data=data, lag=self.estmap_params['lag'],
                                      init_params={'auto_loc': init_loc,
                                                   'auto_scale_tril': init_tril
                                                   }
                                      )

        self.msg_debug("\t Success. Extracting solution")
        BEST_loc, BEST_tril = MAP_SVI_results.params['auto_loc'], MAP_SVI_results.params['auto_scale_tril']

        self.diagnostic_loss_init = MAP_SVI_results.losses

        # ----------------------------------
        # Main Scan

        lags_forscan = self.lags
        if self.reverse: lags_forscan = lags_forscan[::-1]
        l_old = np.inf

        scanned_optima = []
        ELBOS_tosave = []
        diagnostic_hessians = []
        diagnostic_losses = []

        for i, lag in enumerate(lags_forscan):
            print(":" * 23)
            self.msg_run("Scanning at lag=%.2f ..." % lag)

            svi_loop_result = autosvi.run(jax.random.PRNGKey(seed),
                                          self.ELBO_Nsteps,
                                          data=data, lag=lag,
                                          init_params={'auto_loc': BEST_loc,
                                                       'auto_scale_tril': BEST_tril
                                                       },
                                          progress_bar=False
                                          )

            NEW_loc, NEW_tril = svi_loop_result.params['auto_loc'], svi_loop_result.params['auto_scale_tril']

            # --------------
            # Check if the optimization has suceeded or broken

            l_old = l_old
            l_new = svi_loop_result.losses[-1]
            diverged = bool(np.isinf(NEW_loc).any() + np.isinf(NEW_tril).any())

            self.msg_run(
                "From %.2f to %.2f, change of %.2f against %.2f" % (l_old, l_new, l_new - l_old, self.ELBO_threshold))

            if l_new - l_old < self.ELBO_threshold and not diverged:
                self.msg_run("Seems to have converged at iteration %i / %i" % (i, self.Nlags))

                self.converged[i] = True
                l_old = l_new
                BEST_loc, BEST_tril = NEW_loc, NEW_tril

                uncon_params = self.stat_model.to_uncon(self.estmap_params | {'lag': lag}) | _utils.dict_unpack(NEW_loc,
                                                                                                                self.params_toscan)
                con_params = self.stat_model.to_con(uncon_params)
                scanned_optima.append(con_params)

                H = np.dot(NEW_tril, NEW_tril.T)
                H = (H + H.T) / 2
                H = jnp.linalg.inv(H)
                diagnostic_hessians.append(H)

                diagnostic_losses.append(svi_loop_result.losses)
                ELBOS_tosave.append(-1 * svi_loop_result.losses[-int(self.ELBO_Nsteps * self.ELBO_fraction):].mean())

            else:
                self.msg_run("Unable to converge at iteration %i / %i" % (i, self.Nlags))
                self.msg_debug("Reason for failure: \n large ELBO drop: \t %r \n diverged: \t %r" % (
                    l_new - l_old < self.ELBO_threshold, diverged))

        self.ELBOS = np.array(ELBOS_tosave)
        self.diagnostic_losses = np.array(diagnostic_losses)
        self.diagnostic_hessians = np.array(diagnostic_hessians)

        self.msg_run("Scanning Complete. Calculating ELBO integrals...")

        self.scan_peaks = _utils.dict_combine(scanned_optima)

        # ---------------------------------------------------------------------------------
        # For each of these peaks, estimate the evidence
        # todo - vmap and parallelize

        Zs = []
        for j, params in enumerate(scanned_optima):
            Z_ELBO = self.ELBOS[j]
            # if not self.constrained_domain: Z_ELBO += self.stat_model.uncon_grad(params)
            Zs.append(Z_ELBO)

        self.log_evidences = np.array(Zs)
        self.has_run = True

        self.msg_run("SVI Fitting complete.", "-" * 23, "-" * 23, delim='\n')

    def refit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):

        return

    def diagnostics(self, plot=True):

        f, (a2, a1) = plt.subplots(2, 1)

        for i, x in enumerate(self.diagnostic_losses):
            a1.plot(x - (-self.ELBOS[i]), c='k', alpha=0.25)
        a2.plot(self.diagnostic_loss_init, c='k')

        a1.axvline(int((1 - self.ELBO_fraction) * self.ELBO_Nsteps), c='k', ls='--')

        a1.set_yscale('symlog')
        a2.set_yscale('symlog')
        a1.grid(), a2.grid()

        a1.set_xlim(0, self.ELBO_Nsteps)
        a2.set_xlim(0, self.ELBO_Nsteps_init)

        a1.set_title("Scan SVIs")
        a2.set_title("Initial MAP SVI")

        f.supylabel("Loss - loss_final (log scale)")
        a1.set_xlabel("iteration Number"), a2.set_xlabel("iteration Number")

        txt = "Trace plots of ELBO convergence. All lines should be flat by the right hand side.\n" \
              "Top panel is for initial guess and need only be flat. Bottom panel should be flat within" \
              "averaging range, i.e. to the right of dotted line."
        f.supxlabel(r'$\begin{center}X-axis\\*\textit{\small{%s}}\end{center}$' % txt)

        f.tight_layout()

        if plot: plt.show()


# =====================================================
if __name__ == "__main__":
    from mocks import mock_A, mock_B, mock_C

    import matplotlib.pyplot as plt
    from models import dummy_statmodel

    #::::::::::::::::::::

    mock = mock_A(seed=5)
    mock01 = mock.lc_1
    mock02 = mock.lc_2
    lag_true = mock.lag

    plt.figure()
    mock().plot(axis=plt.gca())
    plt.legend()
    plt.grid()
    plt.show()

    #::::::::::::::::::::
    test_statmodel = dummy_statmodel()

    # ICCF Test
    test_ICCF = ICCF(Nboot=128, Nterp=128, Nlags=128, stat_model=test_statmodel)
    print("Doing Fit")
    test_ICCF.fit(mock01, mock02)
    print("Fit done")

    ICCF_samples = test_ICCF.get_samples()['lag']
    print(ICCF_samples.mean(), ICCF_samples.std())

    plt.figure()
    plt.hist(ICCF_samples, histtype='step', bins=24)
    plt.axvline(lag_true, ls='--', c='k', label="True Lag")
    plt.axvline(ICCF_samples.mean(), ls='--', c='r', label="Mean Lag")
    plt.title("ICCF Results")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------
    # Prior Sampling

    test_prior_sampler = prior_sampling(stat_model=test_statmodel)
    test_prior_sampler.fit(lc_1=mock01, lc_2=mock02, seed=0)
    test_samples = test_prior_sampler.get_samples(512, importance_sampling=True)

    plt.figure()
    plt.title("Dummy prior sampling test")
    plt.hist(test_samples['lag'], histtype='step', density=True)
    plt.axvline(250.0, ls='--', c='k')
    plt.axvline(test_samples['lag'].mean(), ls='--', c='r')
    plt.ylabel("Posterior Density")
    plt.xlabel("Lag")
    plt.grid()
    plt.tight_layout()
    plt.show()
