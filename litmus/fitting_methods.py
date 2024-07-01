'''
Contains fitting procedures to be executed by the litmus class object

HM 24
'''

# ============================================
# IMPORTS
import sys

import clustering
import jaxopt
import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from jax.random import PRNGKey

import jax
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax.numpy as jnp
import numpy as np
from numpy import nan

from models import _default_config

import _utils
from functools import partial

from ICCF_working import *

from models import stats_model

from lightcurve import lightcurve


# ============================================

# Base fitting procedure class
class fitting_procedure(object):
    '''
    Generic class for lag fitting procedures. Contains parent methods for setting properties
    '''

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=True, **fit_params):

        if not hasattr(self, "_default_params"):
            self._default_params = {}
        if not hasattr(self, "results"):
            self.results = {}

        self.stat_model = stat_model

        self.debug = debug
        self.verbose = verbose
        self.out_stream = out_stream
        self.err_stream = err_stream

        self.name = "Base Fitting Procedure"

        self.is_ready = False
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
        elif False and key not in ['results'] \
                and hasattr(self, 'results') \
                and key in self.results.keys():
            return (self.results[key])
        else:
            return (super().__getattribute__(key))

    def __setattr__(self, key, value):
        if key not in ["_default_params", "fitting_params"] \
                and hasattr(self, "_default_params") \
                and hasattr(self, "fitting_params") \
                and key in self._default_params.keys():
            self.fitting_params[key] = value
            self.has_run, self.is_ready = False, False
        elif False and key not in ['results'] \
                and hasattr(self, 'results') \
                and key in self.results.keys():
            self.results[key] = value
        else:
            super().__setattr__(key, value)

    # ----------------------

    def reset(self):
        '''
        Clears all memory and resets params to defaults
        '''
        self.set_config(**self._default_params)

        self.has_run, self.is_ready = False, False

        return

    def set_config(self, **fit_params):
        '''
        Configure fitting parameters for fitting_method() object
        Accepts any parameters present with a name in fitting_method.fitting_params
        Unlisted parameters will be ignored.
        '''

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
            print("Tried to configure bad keys:", end="\t", file=self.err_stream)
            for key in badkeys: print(key, end=', ', file=self.err_stream)
            print()

        return

    def readyup(self):
        '''
        Performs pre-fit preparation calcs. Should only be called if not self.is_ready()
        '''
        self.is_ready = True

    # ----------------------
    # Error message printing
    def msg_err(self, *x: str, end='\n', delim=' '):
        '''
        Messages for when something has broken or been called incorrectly
        '''
        if True:
            for a in x:
                print(a, file=self.err_stream, end=delim)

        print(end, end='')
        return

    def msg_run(self, *x: str, end='\n', delim=' '):
        '''
        Standard messages about when things are running
        '''
        if self.verbose:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return

    def msg_verbose(self, *x: str, end='\n', delim=' '):
        '''
        Explicit messages to help debug when things are behaving strangely
        '''
        if self.debug:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return

    # ----------------------
    # Main methods

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        '''
        Fit lags
        :param lc_1: Lightcurve 1 (Main)
        :param lc_2: Lightcurve 2 (Response)
        :param stat_model: a statistical model object
        :param seed: A random seed for feeding to the fitting process. If none, will select randomly
        '''

        # Sanity checks inherited by all sub-classes
        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
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
        '''
        Returns the estimated evidence for the fit model. Returns as array-like [Z,dZ-,dZ+]
        '''

        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_evidence == fitting_procedure.get_evidence:
            self.msg_err("Fitting \"%s\" method does not have method .get_evidence() implemented" % self.name)

        return (np.array([0.0, 0.0, 0.0]))

    def get_information(self, seed: int = None) -> [float, float, float]:
        '''
        Returns an estimate of the information (KL divergence relative to prior). Returns as array-like [I,dI-,dI+]
        '''

        if not self.is_ready: self.readyup()
        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_information == fitting_procedure.get_information:
            self.msg_err("Fitting \"%s\" method does not have method .get_information() implemented" % self.name)

        return (np.array([0.0, 0.0, 0.0]))

    def get_peaks(self, seed=None):
        '''
        Returns the maximum posterior position in parameter space
        '''

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
    '''
    Fit lags using interpolated cross correlation function
    todo
        - Add p value, false positive and evidence estimates
    '''

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        self._default_params = {
            'Nboot': 512,
            'Nterp': 2014,
            'Nlags': 512,
        }

        super().__init__(**args_in)

        self.name = "ICCF Fitting Procedure"
        self.lags = np.zeros(self.Nterp)

        self.results = {'samples': np.zeros(self.Nboot),
                        'correl_curve': np.zeros(self.Nterp),
                        'lag_mean': 0.0,
                        'lag_err': 0.0
                        }

    # -------------------------
    def set_config(self, **fit_params):
        super().set_config(**fit_params)

    def readyup(self):
        super().readyup()
        self.lags = jnp.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags)
        self.is_ready = True

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
        self.results['samples'] = jax_samples
        self.results['lag_mean'] = jax_samples.mean()
        self.results['lag_err'] = jax_samples.std()

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
            return ({'lag': self.results['samples']})
        else:
            if N > self.Nboot:
                self.msg_err(
                    "Warning, tried to get %i sub-samples from %i boot-strap itterations in ICCF" % (N, self.Nboot),
                )
            return ({'lag': np.random.choice(a=self.results['samples'], size=N, replace=True)})

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
    '''
    Randomly samples from the prior and weights with importance sampling.
    The crudest available sampler outside of a gridsearch.
    '''

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        # ------------------------------------
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        self._default_params = {
            'Nsamples': 4096
        }

        super().__init__(**args_in)
        # ------------------------------------

        self.name = "Prior Sampling Fitting Procedure"

        self.results = {'samples': np.zeros(self.Nsamples),
                        'log_likes': np.zeros(self.Nsamples),
                        'weights': np.zeros(self.Nsamples)
                        }

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
        self.results['log_prior'] = log_prior
        self.results['log_likes'] = log_likes
        self.results['log_density'] = log_density
        self.results['samples'] = samples
        self.results['weights'] = likes / likes.sum()

        # Mark as having completed a run
        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
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
            weights = self.results['weights'] / self.results['weights'].sum()
        else:
            weights = None

        I = np.random.choice(a=np.arange(self.Nsamples), size=N, replace=True,
                             p=weights)
        return ({
            key: val[I] for key, val in zip(self.results['samples'].keys(), self.results['samples'].values())
        })

    def get_evidence(self, seed=None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------
        density = np.exp(self.results['log_density'])

        Z = density.mean() * self.stat_model.prior_volume
        uncert = density.std() / np.sqrt(self.Nsamples) * self.stat_model.prior_volume

        return (np.array([Z, -uncert, uncert]))

    def get_information(self, seed: int = None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------
        info_partial = np.random.choice(self.results['log_density'] - self.results['log_prior'], self.Nsamples,
                                        p=self.results['weights'])
        info = info_partial.mean() * self.stat_model.prior_volume
        uncert = info_partial.std() / np.sqrt(self.Nsamples) * self.stat_model.prior_volume

        return (np.array([info, -uncert, uncert]))


# ============================================
# Nested Sampling
class nested_sampling(fitting_procedure):
    '''
    Simple direct nested sampling. Not ideal.
    '''

    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):

        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        self._default_params = {
            'num_live_points': 5000,
            'max_samples': 50000,
            'num_parallel_samplers': 1,
            'uncert_improvement_patience': 2,
            'live_evidence_frac': 0.01,
        }

        super().__init__(**args_in)

        self.name = "Prior Sampling Fitting Procedure"

        self.sampler = None

        self.results = {
            'logevidence': jnp.zeros(3),
            'priorvolume': 0.0,
        }

    # --------------
    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        if seed is None: seed = _utils.randint()

        NS = NestedSampler(self.stat_model,
                           constructor_kwargs={key: self.fitting_params[key]
                                               for key in ['num_live_points',
                                                           'max_samples',
                                                           'num_parallel_samplers',
                                                           'uncert_improvement_patience']
                                               },
                           termination_kwargs={'live_evidence_frac': self.fitting_params['live_evidence_frac']})

        data = self.stat_model.lc_to_data(lc_1, lc_2)
        NS.run(data=data, rng_key=jax.random.PRNGKey(seed))

        # Store Results & Necessary Values
        self.results['priorvolume'] = self.stat_model.prior_volume
        self.results['logevidence'] = np.array(
            [NS._results.log_Z_mean - np.log(self.stat_model.prior_volume), NS._results.log_Z_uncert]
        )

        # Mark good for retrieval
        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        if seed is None: seed = _utils.randint()

        NS = self.sampler

        if not importance_sampling:
            samples, weights = NS.get_weighted_samples()
        else:
            samples = NS.get_samples(jax.random.PRNGKey(seed), N)

        return (samples)

    def get_evidence(self, seed: int = None) -> [float, float, float]:
        '''
        Returns the -1, 0 and +1 sigma values for model evidence from nested sampling.
        This represents an estimate of numerical uncertainty
        '''

        if seed is None: seed = _utils.randint()

        l, l_e = self.results['logevidence']

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
        if seed is None: seed = _utils.randint()

        NS = self.sampler
        samples, logweights = NS.get_weighted_samples()

        weights = np.exp(logweights)
        weights /= weights.sum()

        log_density = NS._results.log_posterior_density
        prior_values = self.stat_model.log_prior(samples)

        info = np.sum((log_density - prior_values) * weights)

        partial_info = np.random.choice((log_density - prior_values), len(log_density), p=weights)
        uncert = partial_info.std() / np.sqrt(len(log_density))

        return (np.array(info, uncert, uncert))

    def get_peaks(self, seed: int = None) -> ({str: [float]}, float):
        if seed is None: seed = _utils.randint()

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


# ------------------------------------------------------
class hessian_scan(fitting_procedure):
    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        self._default_params = {
            'Nlags': 1024,
            'opt_tol': 1E-5,
            'step_size': 0.1,
            'constrained_domain': False,
            'max_opt_eval': 1_000,
            'solvertype': jaxopt.GradientDescent
        }

        super().__init__(**args_in)

        self.name = "Hessian Scan Fitting Procedure"
        self.results = {
            'lags': None,
            'opt_positions': None,
            'opt_densities': None,
            'opt_evidence': None,
            'opt_hessians': None,
            'evidence': None,
        }

    def readyup(self):
        self.lags = np.linspace(*self.stat_model.prior_ranges['lag'], self.Nlags)
        self.is_ready = True

    # --------------

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------
        # todo - break this into sub-functions so other functions can inherit

        # Generate a random starting point
        self.msg_run("Starting Hessian Scan")

        start_params = self.stat_model.prior_sample(seed=seed)
        start_params |= {'lag': self.lags[0]}

        self.msg_run("Beginning scan at:", start_params)

        params_toscan = [key for key in start_params.keys() if key not in ['lag']]

        self.msg_run("Optimizing for parameters:", *params_toscan)
        data = self.stat_model.lc_to_data(lc_1, lc_2)

        # Run over 'self.lags' and scan all positions
        # todo - Change this to call the vmapped version of stat_model.scan()
        # todo - Expand functionality to allow for different optimizers
        scanned_params = []

        for lag in self.lags:
            self.msg_run("Scanning at lag=%.2f ..." % lag)
            start_params = self.stat_model.scan(start_params=start_params,
                                                optim_params=params_toscan,
                                                stepsize=self.step_size,
                                                maxiter=self.max_opt_eval,
                                                tol=self.opt_tol,
                                                data=data
                                                )
            scanned_params.append(start_params.copy())
        self.scan_peaks = _utils.dict_combine(scanned_params)

        # For each of these peaks, estimate the evidence
        # todo - add a max LL significance to cut down on evals
        # todo - vmap and parallelize
        Zs = []
        for params in scanned_params:
            Zs.append(self.stat_model.laplace_log_evidence(params=params, data=data))
        self.evidences = np.array(Zs)

    def get_evidence(self, seed: int = None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_evidence(**locals())
        seed = self._tempseed
        # -------------------
        dlag = self.lags.ptp / (self.Nlags - 1)
        Z = self.evidences.sum() * dlag
        return (np.array(Z, Z, np.inf))


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
