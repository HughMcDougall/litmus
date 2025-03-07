'''
Contains fitting procedures to be executed by the litmus class object

HM 24
'''

import importlib.util
# ============================================
# IMPORTS
import sys
from typing import Callable

import jaxopt
import numpyro
import numpy as np
import matplotlib.pyplot as plt

# ------
# Samplers, stats etc

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
    pass

# ------
# Internal
import litmus._utils as _utils
import litmus.clustering as clustering
from litmus.ICCF_working import *

from litmus.models import quickprior
from litmus.models import stats_model
from litmus.lightcurve import lightcurve
from litmus.lin_scatter import linscatter, expscatter


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
            return self.fitting_params[key]
        else:
            return super().__getattribute__(key)

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
            currval = self.__getattribute__(key)
            if self.has_run and val != currval: self.has_run = False

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

        data = self.stat_model.lc_to_data(lc_1, lc_2)
        self._data = data

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

    def get_evidence(self, seed: int = None, return_type='linear') -> [float, float, float]:
        """
        Returns the estimated evidence for the fit model.
        if return_type = 'linear', returns as array-like [Z,-dZ-,dZ+]
        if return_type = 'log', returns as array-like [logZ,-dlogZ-,dlogZ+]
        """

        assert return_type in ['linear', 'log'], "Return type must be 'linear' or 'log'"

        if not self.is_ready: self.readyup()
        if not self.has_run: self.msg_err("Warning! Tried to call get_evidence without running first!")

        if isinstance(seed, int):
            self._tempseed = seed
            self._tempseed = _utils.randint()
        seed = self._tempseed

        if self.__class__.get_evidence == fitting_procedure.get_evidence:
            self.msg_err("Fitting \"%s\" method does not have method .get_evidence() implemented" % self.name)
            return np.array([0.0, 0.0, 0.0])

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

            return np.array([0.0, 0.0, 0.0])

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

            return {}, np.array([])


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
            self._default_params = {}

        self._default_params |= {
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
            self._default_params = {}
        self._default_params |= {
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

    def get_evidence(self, seed=None, return_type='linear') -> [float, float, float]:
        # -------------------
        fitting_procedure.get_evidence(**locals())
        seed = self._tempseed
        # -------------------
        density = np.exp(self.log_density - self.log_density.max())

        Z = density.mean() * self.stat_model.prior_volume * np.exp(self.log_density.max())
        uncert = density.std() / np.sqrt(self.Nsamples) * self.stat_model.prior_volume

        if return_type == 'linear':
            np.array([Z, -uncert, uncert])
        elif return_type == 'log':
            np.array([np.log(Z), np.log(1 - uncert / Z), np.log(1 + uncert / Z)])

    def get_information(self, seed: int = None) -> [float, float, float]:
        # -------------------
        fitting_procedure.get_information(**locals())
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
            self._default_params = {}
        self._default_params |= {
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

    def get_evidence(self, seed: int = None, return_type='linear') -> [float, float, float]:
        '''
        Returns the -1, 0 and +1 sigma values for model evidence from nested sampling.
        This represents an estimate of numerical uncertainty
        '''

        if seed is None: seed = _utils.randint()

        l, l_e = self._jaxnsresults.log_Z_mean, self._jaxnsresults.log_Z_uncert

        if return_type == 'linear':

            out = np.exp([
                l,
                l - l_e,
                l + l_e
            ])

            out -= np.array([0, out[0], out[0]])
        elif return_type == 'log':
            out = np.array([l, -l_e, l_e])

        return out

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
            self._default_params = {}

        self._default_params |= {
            'Nlags': 64,
            'opt_tol': 1E-2,
            'opt_tol_init': 1E-4,
            'step_size': 0.001,
            'constrained_domain': False,
            'max_opt_eval': 1_000,
            'max_opt_eval_init': 5_000,
            'LL_threshold': 100.0,
            'init_samples': 5_000,
            'grid_bunching': 0.5,
            'grid_depth': None,
            'grid_Nterp': None,
            'grid_relaxation': 0.1,  # deprecated, remove
            'grid_firstdepth': 2.0,
            'reverse': True,
            'split_lags': True,
            'optimizer_args_init': {},
            'optimizer_args': {},
            'seed_params': {},
            'precondition': 'diag',
            'interp_scale': 'log',
        }

        self._allowable_interpscales = ['linear', 'log']

        super().__init__(**args_in)

        # -----------------------------------

        self.name = "Hessian Scan Fitting Procedure"

        self.lags: [float] = np.zeros(self.Nlags)
        self.converged: np.ndarray[bool] = np.zeros_like(self.lags, dtype=bool)

        self.scan_peaks: dict = {None}
        self.log_evidences: list = None
        self.log_evidences_uncert: list = None

        self.diagnostic_hessians: list = None
        self.diagnostic_densities: list = None
        self.diagnostic_grads: list = None
        self.diagnostic_ints: list = None
        self.diagnostic_tgrads: list = None

        self.params_toscan = self.stat_model.free_params()
        if 'lag' in self.params_toscan: self.params_toscan.remove('lag')

        self.precon_matrix: np.ndarray[np.float64] = np.eye(len(self.params_toscan), dtype=np.float64)
        self.solver: jaxopt.BFGS = None

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
        self.diagnostic_densities = []
        self.log_evidences_uncert = []

        self.params_toscan = [key for key in self.stat_model.paramnames() if
                              key not in ['lag'] and key in self.stat_model.free_params()
                              ]

        self.is_ready = True

    # --------------
    # Setup Funcs

    def estimate_MAP(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        '''
        :param lc_1:
        :param lc_2:
        :param seed:
        :return:
        '''

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

        self.msg_run("Moving non-lag params to new location...")
        estmap_params = self.stat_model.scan(start_params=seed_params,
                                             optim_params=[key for key in self.stat_model.free_params() if
                                                           key != 'lag'],
                                             data=data,
                                             optim_kwargs=self.optimizer_args_init,
                                             precondition=self.precondition
                                             )
        ll_firstscan = self.stat_model.log_density(estmap_params, data)
        if 'lag' in self.stat_model.free_params():
            self.msg_run("Optimizer settled at new fit:")
            for it in estmap_params.items():
                self.msg_run('\t %s: \t %.2f' % (it[0], it[1]))
            self.msg_run(
                "Log-Density for this is: %.2f" % ll_firstscan
            )

            self.msg_run("Finding a good lag...")
            test_lags = self.stat_model.prior_sample(self.init_samples)['lag']
            test_samples = _utils.dict_extend(estmap_params, {'lag': test_lags})
            ll_test = self.stat_model.log_density(test_samples, data)
            bestlag = test_lags[ll_test.argmax()]

            self.msg_run("Grid finds good lag at %.2f:" % bestlag)
            self.msg_run(
                "Log-Density for this is: %.2f" % ll_test.max()
            )

            if ll_test.max() > ll_firstscan:
                bestlag = bestlag
            else:
                bestlag = estmap_params['lag']

            estmap_params = self.stat_model.scan(start_params=estmap_params | {'lag': bestlag},
                                                 optim_params=['lag'],
                                                 data=data,
                                                 optim_kwargs=self.optimizer_args_init,
                                                 precondition=self.precondition
                                                 )

            self.msg_run("Lag-only opt settled at new lag %.2f..." % estmap_params['lag'])

        ll_end = self.stat_model.log_density(estmap_params,
                                             data=data
                                             )

        # ----------------------------------
        # CHECKING OUTPUTS

        self.msg_run("Optimizer settled at new fit:")
        for it in estmap_params.items():
            self.msg_run('\t %s: \t %.2f' % (it[0], it[1]))
        self.msg_run(
            "Log-Density for this is: %.2f" % ll_end
        )

        # ----------------------------------
        # CHECKING OUTPUTS

        if ll_end < ll_start:
            self.msg_err("Warning! Optimization seems to have diverged. Defaulting to seed params. \n"
                         "Please consider running with different optim_init inputs")
            estmap_params = seed_params
        return estmap_params

    def make_grid(self, data, seed_params=None, interp_scale='log'):
        """
        :param data:
        :param seed_params:
        :param interp_scale:
        :return:
        """
        # todo - reformat this and the make_grid for better consistency in drawing from estmap and seed params

        assert interp_scale in ['log', 'linear'], "Interp scale was %s, must be in 'log' or 'linear'" % interp_scale

        if not self.is_ready: self.readyup()

        if 'lag' in self.stat_model.fixed_params():
            lags = np.array([np.mean(self.stat_model.prior_ranges['lag'])])
            self.Nlags = 1
            self.readyup()
            return lags

        # If no seed parameters specified, use stored
        if seed_params is None:
            seed_params = self.estmap_params

        # If these params are incomplete, use find_seed to complete them
        if seed_params.keys() != self.stat_model.paramnames():
            seed_params, llstart = self.stat_model.find_seed(data, guesses=self.init_samples, fixed=seed_params)

        lags = np.linspace(*self.stat_model.prior_ranges['lag'], int(self.Nlags * self.grid_firstdepth) + 1,
                           endpoint=False)[1:]
        lag_terp = np.linspace(*self.stat_model.prior_ranges['lag'], self.grid_Nterp)

        log_density_all, lags_all = np.empty(shape=(1,)), np.empty(shape=(1,))
        for i in range(self.grid_depth):
            params = _utils.dict_extend(seed_params, {'lag': lags})
            log_density_all = np.concatenate([log_density_all, self.stat_model.log_density(params, data)])
            lags_all = np.concatenate([lags_all, lags])
            I = lags_all.argsort()
            log_density_all, lags_all = log_density_all[I], lags_all[I]

            if interp_scale == 'linear':

                density = np.exp(log_density_all - log_density_all.max())

                # Linearly interpolate the density profile
                density_terp = np.interp(lag_terp, lags_all, density, left=0, right=0)
                density_terp /= density_terp.sum()


            elif interp_scale == 'log':

                density = np.exp(log_density_all - log_density_all.max())

                # Linearly interpolate the density profile
                log_density_terp = np.interp(lag_terp, log_density_all - log_density_all.max(), density, left=log_density_all[0], right=log_density_all[-1])
                density_terp = np.exp(log_density_terp)
                density_terp /= density_terp.sum()

            gets = np.linspace(0, 1, self.grid_Nterp)
            percentiles = np.cumsum(density_terp) * self.grid_bunching + gets * (1 - self.grid_bunching)
            percentiles /= percentiles.max()

            lags = np.interp(np.linspace(0, 1, self.Nlags), percentiles, lag_terp,
                             left=lag_terp.min(),
                             right=lag_terp.max()
                             )

        return lags

    # --------------
    # Fiting Funcs

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
        if self.split_lags:
            split_index = abs(lags - self.estmap_params['lag']).argmin()
            lags_left, lags_right = lags[:split_index], lags[split_index:]
            lags = np.concatenate([lags_right, lags_left[::-1]])
            if self.reverse: lags = lags = np.concatenate([lags_left[::-1], lags_right])
        elif self.reverse:
            lags = lags[::-1]
        self.lags = lags

        self.has_prefit = True

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------
        # Setup + prefit if not run
        self.msg_run("Starting Hessian Scan")

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        if not self.has_prefit:
            self.prefit(lc_1, lc_2, seed)
        best_params = self.estmap_params.copy()

        # ----------------------------------
        # Create scanner and perform setup
        params_toscan = self.params_toscan
        lags_forscan = self.lags.copy()

        solver, runsolver, [converter, deconverter, optfunc, runsolver_jit] = self.stat_model._scanner(data,
                                                                                                       optim_params=params_toscan,
                                                                                                       optim_kwargs=self.optimizer_args,
                                                                                                       return_aux=True
                                                                                                       )
        self.solver = solver
        x0, y0 = converter(best_params)
        state = solver.init_state(x0, y0, data)

        # ----------------------------------
        # Sweep over lags
        scanned_optima, grads, Hs = [], [], []
        tols, Zs, Ints, tgrads = [], [], [], []
        for i, lag in enumerate(lags_forscan):
            self.msg_run(":" * 23)
            self.msg_run("Scanning at lag=%.2f ..." % lag)

            # Get current param site in packed-function friendly terms
            opt_params, aux_data, state = runsolver_jit(solver, best_params | {'lag': lag}, state)

            # --------------
            # Check if the optimization has succeeded or broken

            l_1 = self.stat_model.log_density(best_params | {'lag': lag}, data)
            l_2 = self.stat_model.log_density(opt_params | {'lag': lag}, data)
            bigdrop = l_2 - l_1 < -self.LL_threshold
            diverged = np.any(np.isinf(np.array([x for x in self.stat_model.to_uncon(opt_params).values()])))

            self.msg_run("Change of %.2f against %.2f" % (l_2 - l_1, self.LL_threshold))

            if not bigdrop and not diverged:
                self.converged[i] = True

                is_good = [True, True, True]

                # ======
                # Check position & Grad
                try:
                    uncon_params = self.stat_model.to_uncon(opt_params)
                    log_height = self.stat_model.log_density_uncon(uncon_params, data)
                except:
                    self.msg_err("Something wrong!")
                    is_good[0] = False

                # ======
                # Check tolerances & hessians
                try:
                    H = self.stat_model.log_density_uncon_hess(uncon_params, data, keys=params_toscan)
                    assert np.linalg.det(H), "Error in H calc"
                    tol = self.stat_model.opt_tol(opt_params, data, integrate_axes=params_toscan)
                except:
                    self.msg_err("Something wrong in Hessian / Tolerance!:")
                    is_good[1] = False

                # ======
                # Get evidence
                try:
                    laplace_int = self.stat_model.laplace_log_evidence(opt_params, data,
                                                                       integrate_axes=params_toscan,
                                                                       constrained=self.constrained_domain)
                    tgrad = self.stat_model.uncon_grad_lag(opt_params) if not self.constrained_domain else 0
                    Z = laplace_int + tgrad
                    assert not np.isnan(Z), "Error in Z calc"
                except:
                    self.msg_err("Something wrong in Evidence!:")
                    is_good[2] = False

                # Check and save if good
                if np.all(is_good):
                    self.msg_run(
                        "Seems to have converged at iteration %i / %i with tolerance %.2e" % (i, self.Nlags, tol))

                    if tol < 1.0:
                        best_params = opt_params
                    else:
                        self.msg_run("Possibly stuck in a furrow. Resetting start params")
                        best_params = self.estmap_params.copy()

                    scanned_optima.append(opt_params.copy())
                    tols.append(tol)

                    grads.append(aux_data['grad'])
                    Hs.append(H)
                    Ints.append(laplace_int)
                    tgrads.append(tgrad)
                    Zs.append(Z)
                else:
                    self.msg_run("Seems to have severely diverged at iteration %i / %i" % (i, self.Nlags))
                    reason = ["Eval", "hessian/tol", "evidence"]
                    for a, b in zip(reason, is_good):
                        self.msg_run("%s:\t%r" % (a, b))

            else:
                self.converged[i] = False
                self.msg_run("Unable to converge at iteration %i / %i" % (i, self.Nlags),
                             "\nLarge Drop?:\t", bigdrop,
                             "\nOptimizer Diverged:\t", diverged)

        if sum(self.converged) == 0:
            self.msg_err("All slices catastrophically diverged! Try different starting conditions and/or grid spacing")

        self.msg_run("Scanning Complete. Calculating laplace integrals...")

        # --------
        # Save and apply
        self.diagnostic_grads = grads
        self.diagnostic_hessians = Hs
        self.diagnostic_tgrads = np.array(tgrads).squeeze().flatten()
        self.diagnostic_ints = np.array(Ints).squeeze().flatten()

        self.scan_peaks = _utils.dict_combine(scanned_optima)
        self.diagnostic_densities = self.stat_model.log_density(self.scan_peaks, data)
        self.log_evidences = np.array(Zs).squeeze().flatten()
        self.log_evidences_uncert = np.square(tols).squeeze().flatten()

        self.msg_run("Hessian Scan Fitting complete.", "-" * 23, "-" * 23, delim='\n')
        self.has_run = True

    def refit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # -------------------
        fitting_procedure.fit(**locals())
        seed = self._tempseed
        # -------------------

        data = self.stat_model.lc_to_data(lc_1, lc_2)

        peaks = _utils.dict_divide(self.scan_peaks)
        I = np.arange(len(peaks))
        select = np.argwhere(self.log_evidences_uncert > self.opt_tol).squeeze()
        if not (_utils.isiter(select)): select = np.array([select])

        peaks, I = np.array(peaks)[select], I[select]

        self.msg_run("Doing re-fitting of %i lags" % len(peaks))
        newtols = []
        for j, i, peak in zip(range(len(I)), I, peaks):

            self.msg_run(":" * 23, "Refitting lag %i/%i at lag %.2f" % (j, len(peaks), peak['lag']), delim='\n')

            ll_old = self.stat_model.log_density(peak, data)
            old_tol = self.log_evidences_uncert[i]

            new_peak = self.stat_model.scan(start_params=peak,
                                            optim_params=self.params_toscan,
                                            data=data,
                                            optim_kwargs=self.optimizer_args,
                                            precondition=self.precondition
                                            )
            ll_new = self.stat_model.log_density(new_peak, data)
            if ll_old > ll_new or np.isnan(ll_new):
                self.msg_err("New peak bad (LL from %.2e to %.2e. Trying new start location" % (ll_old, ll_new))
                new_peak = self.stat_model.scan(start_params=self.estmap_params,
                                                optim_params=self.params_toscan,
                                                data=data,
                                                optim_kwargs=self.optimizer_args,
                                                precondition=self.precondition
                                                )
                ll_new = self.stat_model.log_density(new_peak, data)

                if ll_old > ll_new:
                    self.msg_err("New peak bad (LL from %.2e to %.2e. Trying new start location" % (ll_old, ll_new))
                    continue

            new_peak_uncon = self.stat_model.to_uncon(new_peak)
            new_grad = _utils.dict_pack(new_grad, keys=self.params_toscan)
            new_hessian = self.stat_model.log_density_uncon_hess(new_peak_uncon, data, keys=self.params_toscan)

            try:
                int = self.stat_model.laplace_log_evidence(new_peak, data, constrained=self.constrained_domain)
                tgrad = self.stat_model.uncon_grad_lag(new_peak)
                Z = tgrad + int
                Hinv = np.linalg.inv(new_hessian)
            except:
                self.msg_run("Optimization failed on %i/%i" % (j, len(peaks)))
                continue

            tol = self.stat_model.opt_tol(new_peak, data, self.params_toscan)

            if tol < old_tol:
                self.diagnostic_tgrads[i] = tgrad
                self.diagnostic_ints[i] = int
                self.log_evidences[i] = tgrad + int

                self.diagnostic_grads[i] = new_grad
                self.diagnostic_hessians[i] = new_hessian
                self.log_evidences_uncert[i] = tol ** 2
                self.msg_run("Settled at new tol %.2e" % tol)
            else:
                self.msg_run(
                    "Something went wrong at this refit! Consider changing the optimizer_args and trying again")
        self.msg_run("Refitting complete.")

    # --------------
    # Checks

    def diagnostics(self, plot=True):
        '''
        Runs some diagnostics for convergence
        :return:
        '''

        loss = self.log_evidences_uncert

        lagplot = self.scan_peaks['lag']
        I = self.scan_peaks['lag'].argsort()
        lagplot = lagplot[I]
        Y = self.estmap_tol[I]

        # ---------
        fig = plt.figure()
        plt.ylabel("Loss Norm, $ \\vert \Delta x / \sigma_x \\vert$")
        plt.xlabel("Scan Lag No.")
        plt.plot(lagplot, loss, 'o-', c='k', label="Scan Losses")
        plt.scatter(self.estmap_params['lag'], Y, c='r', marker='x', s=40, label="Initial MAP Scan Loss")
        plt.axhline(self.opt_tol, ls='--', c='k', label="Nominal Tolerance Limit")
        plt.legend(loc='best')

        fig.text(.5, .05, "How far each optimization slice is from its peak. Lower is good.", ha='center')
        plt.yscale('log')
        plt.grid()
        plt.show()

    def diagnostic_lagplot(self, show=True):
        f, (a1, a2) = plt.subplots(2, 1, sharex=True)

        lags_forint, logZ_forint, density_forint = self._get_slices('lags', 'logZ', 'densities')

        # ---------------------
        for a in (a1, a2):
            a.scatter(lags_forint, np.exp(logZ_forint - logZ_forint.max()), label="Evidence")
            a.scatter(lags_forint, np.exp(density_forint - density_forint.max()), label="Density")
            a.grid()

        a2.set_yscale('log')
        a1.legend()

        # --------------
        # Outputs
        if show: plt.show()
        return f

    def _get_slices(self, *args):
        '''
        Summarizes the currently good scan peaks & lag slices
        Combined in one function for utility
            'lags': lags_forint,
            'logZ': logZ_forint,
            'dlogZ': logZ_uncert_forint,
            'peaks': peaks,
            'covars': covars,
            'densities': densities,
            'grads': grads,
        '''

        good_tol = self.log_evidences_uncert <= self.opt_tol
        good_tgrad = abs(self.diagnostic_tgrads) <= np.median(abs(self.diagnostic_tgrads)) * 10
        select = np.argwhere(good_tol * good_tgrad).squeeze()
        if not (_utils.isiter(select)): select = np.array([select])
        if len(select) == 0:
            self.msg_err("High uncertainty in slice evidences: result may be innacurate!. Try re-fitting.")
            select = np.where(good_tgrad)[0]

        # Calculating Evidence
        select = select[self.scan_peaks['lag'][select].argsort()]
        lags_forint = self.scan_peaks['lag'][select]
        logZ_forint = self.log_evidences[select]
        logZ_uncert_forint = self.log_evidences_uncert[select]

        grads = np.array([self.diagnostic_grads[i] for i in select]) if len(self.diagnostic_grads) > 0 else np.zeros(
            len(select))
        densities = self.diagnostic_densities[select]

        peaks = {key: val[select] for key, val in self.scan_peaks.items()}
        peaks = np.array(_utils.dict_divide(peaks))
        covars = -1 * np.array([np.linalg.inv(self.diagnostic_hessians[i]) for i in select])

        out = {
            'lags': lags_forint,
            'logZ': logZ_forint,
            'dlogZ': logZ_uncert_forint,
            'peaks': peaks,
            'covars': covars,
            'densities': densities,
            'grads': grads,
        }

        if args is None:
            return out
        else:
            return (out[key] for key in args)

    def get_evidence(self, seed: int = None, return_type='linear') -> [float, float, float]:
        # -------------------
        fitting_procedure.get_evidence(**locals())
        seed = self._tempseed
        # -------------------

        assert self.interp_scale in self._allowable_interpscales, "Interp scale %s not recognised. Must be selection from %s" % (
            self.interp_scale, self._allowable_interpscales)

        lags_forint, logZ_forint, logZ_uncert_forint = self._get_slices('lags', 'logZ', 'dlogZ')
        minlag, maxlag = self.stat_model.prior_ranges['lag']

        if maxlag - minlag == 0:
            Z = np.exp(logZ_forint.max())
            imax = logZ_forint.argmax()
            uncert_plus, uncert_minus = logZ_uncert_forint[imax], logZ_uncert_forint[imax]


        else:

            if self.interp_scale == 'linear':
                dlag = [*np.diff(lags_forint) / 2, 0]
                dlag[1:] += np.diff(lags_forint) / 2
                dlag[0] += lags_forint.min() - minlag
                dlag[-1] += maxlag - lags_forint.max()

                dlogZ = logZ_forint + np.log(dlag)
                dZ = np.exp(dlogZ - dlogZ.max())
                Z = dZ.sum() * np.exp(dlogZ.max())

                # -------------------------------------
                # Get Uncertainties

                # todo Fix this to be generic and move outside of scope
                # Estimate uncertainty from ~dt^2 error scaling
                dlag_sub = [*np.diff(lags_forint[::2]) / 2, 0]
                dlag_sub[1:] += np.diff(lags_forint[::2]) / 2
                dlag_sub[0] += lags_forint.min() - minlag
                dlag_sub[-1] += maxlag - lags_forint.max()

                dlogZ_sub = logZ_forint[::2] + np.log(dlag_sub)
                dZ_sub = np.exp(dlogZ_sub - dlogZ_sub.max())
                Z_subsample = dZ_sub.sum() * np.exp(dlogZ_sub.max())
                uncert_numeric = abs(Z - Z_subsample) / np.sqrt(17)

                uncert_tol = np.square(dZ * logZ_uncert_forint).sum()
                uncert_tol = np.sqrt(uncert_tol)
                uncert_tol *= np.exp(dlogZ.max())


            elif self.interp_scale == 'log':
                # dZ = dXdY/dln|Y|
                dlag = np.diff(lags_forint)
                dY = np.diff(np.exp(logZ_forint - logZ_forint.max()))
                dE = np.diff(logZ_forint)
                dZ = dlag * dY / dE
                Z = np.sum(dZ) * np.exp(logZ_forint.max())

                uncert_tol = 4 * np.square(
                    np.exp(logZ_forint - logZ_forint.max())[:-1] - dZ / dE
                ) * logZ_uncert_forint[:-1]
                uncert_tol += np.square(logZ_uncert_forint[-1] * np.exp(logZ_forint - logZ_forint.max())[-1])
                uncert_tol = np.sqrt(uncert_tol.sum())
                uncert_tol *= np.exp(logZ_forint.max())

                dlag_sub = np.diff(lags_forint[::2])
                dY_sub = np.diff(np.exp(logZ_forint[::2] - logZ_forint.max()))
                dE_sub = np.diff(logZ_forint[::2])
                dZ_sub = dlag_sub * dY_sub / dE_sub
                Z_subsample = np.sum(dZ_sub) * np.exp(logZ_forint.max())
                uncert_numeric = abs(Z - Z_subsample) / np.sqrt(17)

            self.msg_debug("Evidence Est: \t %.2e" % Z)
            self.msg_debug(
                "Evidence uncerts: \n Numeric: \t %.2e \n Convergence: \t %.2e" % (uncert_numeric, uncert_tol))

            uncert_plus = uncert_numeric + uncert_tol.sum()
            uncert_minus = uncert_numeric

        if return_type == 'linear':
            return np.array([Z, -uncert_minus, uncert_plus])
        elif return_type == 'log':
            return np.array([np.log(Z), np.log(1 - uncert_minus / Z), np.log(1 + uncert_plus / Z)])

    def get_samples(self, N: int = 1, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        # -------------------
        fitting_procedure.get_samples(**locals())
        seed = self._tempseed
        # -------------------

        assert self.interp_scale in self._allowable_interpscales, "Interp scale %s not recognised. Must be selection from %s" % (
            self.interp_scale, self._allowable_interpscales)
        lags_forint, logZ_forint, peaks, covars = self._get_slices('lags', 'logZ', 'peaks', 'covars')

        if len(lags_forint) == 0:
            self.msg_err("Zero good slices in evidence integral!")
            return (0, -np.inf, np.inf)

        # Get weights and peaks etc
        Npeaks = len(lags_forint)
        minlag, maxlag = self.stat_model.prior_ranges['lag']

        Y = np.exp(logZ_forint - logZ_forint.max()).squeeze()

        dlag = [*np.diff(lags_forint) / 2, 0]
        dlag[1:] += np.diff(lags_forint) / 2
        dlag[0] += lags_forint.min() - minlag
        dlag[-1] += maxlag - lags_forint.max()

        if sum(dlag) == 0:
            dlag = 1.0

        weights = Y * dlag
        weights /= weights.sum()

        # Get hessians and peak locations
        if Npeaks > 1:
            I = np.random.choice(range(Npeaks), N, replace=True, p=weights)
        else:
            I = np.zeros(N)

        to_choose = [(I == i).sum() for i in range(Npeaks)]  # number of samples to draw from peak i

        # Sweep over scan peaks and add scatter
        outs = []
        for i in range(Npeaks):
            if to_choose[i] > 0:
                peak_uncon = self.stat_model.to_uncon(peaks[i])

                # Get normal dist properties in uncon space in vector form
                mu = _utils.dict_pack(peak_uncon, keys=self.params_toscan)
                cov = covars[i]

                # Generate samples
                samps = np.random.multivariate_normal(mean=mu, cov=cov, size=to_choose[i])
                samps = _utils.dict_unpack(samps.T, keys=self.params_toscan, recursive=False)
                samps = _utils.dict_extend(peak_uncon, samps)

                # Reconvert to constrained space
                samps = self.stat_model.to_con(samps)

                # -------------
                # Add linear interpolation 'smudging' to lags
                if Npeaks > 1 and 'lag' in self.stat_model.free_params():

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
                    if self.interp_scale == 'linear':
                        tshift = linscatter(Ti, Yi, N=to_choose[i])
                    elif self.interp_scale == 'log':
                        tshift = expscatter(Ti, Yi, N=to_choose[i])
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
            self._default_params = {}

        self._default_params |= {
            'ELBO_threshold': 100.0,
            'ELBO_optimstep': 5E-3,
            'ELBO_particles': 128,
            'ELBO_Nsteps': 128,
            'ELBO_Nsteps_init': 1_000,
            'ELBO_fraction': 0.25,
        }

        super().__init__(**args_in)

        # -----------------------------

        self.name = "SVI Scan Fitting Procedure"

        self.diagnostic_losses = []
        self.diagnostic_loss_init = []
        self.diagnostic_ints = []

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
        # Estimate the MAP and its hessian for starting conditions

        estmap_uncon = self.stat_model.to_uncon(self.estmap_params)

        fix_param_dict_con = {key: self.estmap_params[key] for key in self.stat_model.fixed_params()}
        fix_param_dict_uncon = {key: estmap_uncon[key] for key in self.stat_model.fixed_params()}

        init_hess = -1 * self.stat_model.log_density_uncon_hess(estmap_uncon, data=data, keys=self.params_toscan)

        # Convert these into SVI friendly objects and fit an SVI at the map
        self.msg_run("Performing SVI slice at the MAP estimate")
        init_loc = _utils.dict_pack(estmap_uncon, keys=self.params_toscan)
        init_tril = jnp.linalg.cholesky(jnp.linalg.inv(init_hess))

        bad_starts = False
        if np.isnan(init_loc).any() or np.isnan(init_tril).any():
            self.msg_err("Issue with finding initial solver state for SVI. Proceeding /w numpyro defaults")
            bad_starts = True
        # ----------------------------------
        self.msg_debug("\t Constructing slice model")

        def slice_function(data, lag):
            '''
            This is the conditional model that SVI will map
            '''

            params = {}
            for key in self.stat_model.free_params():
                if key != 'lag':
                    val = quickprior(self.stat_model, key)
                    params |= {key: val}
            params |= {'lag': lag}
            params |= fix_param_dict_con

            with numpyro.handlers.block(hide=self.stat_model.paramnames()):
                LL = self.stat_model._log_likelihood(params, data)

            dilute = -np.log(self.stat_model.prior_ranges['lag'][1] - self.stat_model.prior_ranges['lag'][
                0]) if 'lag' in self.stat_model.free_params() else 0.0
            numpyro.factor('lag_prior', dilute)

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
                                                   } if not bad_starts else None
                                      )

        self.msg_debug("\t Success. Extracting solution")
        BEST_loc, BEST_tril = MAP_SVI_results.params['auto_loc'], MAP_SVI_results.params['auto_scale_tril']

        self.diagnostic_loss_init = MAP_SVI_results.losses

        # ----------------------------------
        # Main Scan

        lags_forscan = self.lags
        l_old = -np.inf

        scanned_optima = []
        ELBOS_tosave = []
        ElBOS_uncert = []
        diagnostic_hessians = []
        diagnostic_losses = []

        for i, lag in enumerate(lags_forscan):
            print(":" * 23)
            self.msg_run("Doing SVI fit at lag=%.2f ..." % lag)

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
            l_new = self._getELBO(svi_loop_result.losses)[0]
            diverged = bool(np.isinf(NEW_loc).any() + np.isinf(NEW_tril).any())
            big_drop = l_new - l_old < - self.ELBO_threshold

            self.msg_run(
                "From %.2f to %.2f, change of %.2f against %.2f" % (l_old, l_new, l_new - l_old, self.ELBO_threshold))

            if not big_drop and not diverged:
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
                H = jnp.linalg.inv(-H)
                diagnostic_hessians.append(H)

                diagnostic_losses.append(svi_loop_result.losses)

                ELBO, uncert = self._getELBO(svi_loop_result.losses)
                ELBOS_tosave.append(ELBO)
                ElBOS_uncert.append(uncert)


            else:
                self.msg_run("Unable to converge at iteration %i / %i" % (i, self.Nlags))
                self.msg_debug("Reason for failure: \n large ELBO drop: \t %r \n diverged: \t %r" % (
                    big_drop, diverged))

        self.msg_run("Scanning Complete. Calculating ELBO integrals...")
        if sum(self.converged) == 0:
            self.msg_err("All slices catastrophically diverged! Try different starting conditions and/or grid spacing")

        self.diagnostic_ints = np.array(ELBOS_tosave)

        self.log_evidences_uncert = np.array(ElBOS_uncert)
        self.diagnostic_losses = np.array(diagnostic_losses)
        self.diagnostic_hessians = np.array(diagnostic_hessians)

        self.scan_peaks = _utils.dict_combine(scanned_optima)
        self.diagnostic_densities = self.stat_model.log_density(self.scan_peaks, data)

        # ---------------------------------------------------------------------------------
        # For each of these peaks, estimate the evidence
        # todo - vmap and parallelize

        Zs, tgrads = [], []
        for j, params in enumerate(scanned_optima):
            Z = self.diagnostic_ints[j]
            tgrad = self.stat_model.uncon_grad_lag(params) if not self.constrained_domain else 0
            tgrad = 0
            tgrads.append(tgrad)
            Zs.append(Z + tgrad)

        self.log_evidences = np.array(Zs)
        self.diagnostic_tgrads = np.array(tgrads)
        self.has_run = True

        self.msg_run("SVI Fitting complete.", "-" * 23, "-" * 23, delim='\n')

    def refit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        # TODO - fill this out

        return

    def _getELBO(self, losses):
        """
        A utility for taking the chains of losses output by SVI and returning
        estimates of log|Z| and dlog|Z|
        """

        N = int(self.ELBO_Nsteps * self.ELBO_fraction)
        ns = np.arange(2, N // 2) * 2
        MEANS = np.zeros(len(ns))
        UNCERTS = np.zeros(len(ns))

        for i, n in enumerate(ns):
            ELBO_samps = -1 * losses[-n:]
            mean = ELBO_samps.mean()

            samps_left, samps_right = np.split(ELBO_samps, 2)
            uncert = ELBO_samps.var() / n
            gap = max(0, samps_left.mean() - samps_right.mean())
            skew = gap ** 2
            uncert += skew

            MEANS[i], UNCERTS[i] = mean, uncert ** 0.5

        mean, uncert = MEANS[UNCERTS.argmin()], UNCERTS.min()

        return (mean, uncert)

    def diagnostics(self, plot=True):

        f, (a2, a1) = plt.subplots(2, 1)

        for i, x in enumerate(self.diagnostic_losses):
            a1.plot(x - (self.diagnostic_ints[i]), c='k', alpha=0.25)
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
        f.supxlabel('$\begin{center}X-axis\\*\textit{\small{%s}}\end{center}$' % txt)

        f.tight_layout()

        if plot: plt.show()


# ------------------------------------------------------
class JAVELIKE(fitting_procedure):
    def __init__(self, stat_model: stats_model,
                 out_stream=sys.stdout, err_stream=sys.stderr,
                 verbose=True, debug=False, **fit_params):
        args_in = {**locals(), **fit_params}
        del args_in['self']
        del args_in['__class__']
        del args_in['fit_params']

        if not hasattr(self, '_default_params'):
            self._default_params = {}
        self._default_params |= {
            'alpha': 2.0,
            'num_chains': 256,
            'num_samples': 200_000 // 256,
            'num_warmup': 5_000,
        }

        self.sampler: numpyro.infer.MCMC = None
        self.kernel: numpyro.infer.AEIS = None
        self.limited_model: Callable = None

        super().__init__(**args_in)

        # -----------------------------

        self.name = "AEIS JAVELIN Emulator fitting Procedure"

    def readyup(self):

        fixed_vals = {key: self.stat_model.prior_ranges[key][0] for key in self.stat_model.fixed_params()}

        def limited_model(data):
            with numpyro.handlers.block(hide=self.stat_model.fixed_params()):
                params = {key: val for key, val in zip(self.stat_model.paramnames(), self.stat_model.prior())}

            params |= fixed_vals
            with numpyro.handlers.block(hide=self.stat_model.paramnames()):
                LL = self.stat_model._log_density(params, data)

            numpyro.factor('ll', LL)

        self.limited_model = limited_model

        self.kernel = numpyro.infer.AIES(self.limited_model,
                                         moves={numpyro.infer.AIES.StretchMove(a=self.alpha): 1.0}
                                         )

        self.sampler = numpyro.infer.MCMC(self.kernel,
                                          num_warmup=self.num_warmup,
                                          num_samples=self.num_samples,
                                          num_chains=self.num_chains,
                                          chain_method='vectorized',
                                          progress_bar=self.verbose)

        self.is_ready = True

    def prefit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        if seed is None: seed = self.seed
        if not self.is_ready: self.readyup()

        self.msg_run("Running warmup with %i chains and %i samples" % (self.num_chains, self.num_warmup))
        # self.sampler.warmup(jax.random.PRNGKey(seed), self.stat_model.lc_to_data(lc_1, lc_2))

        self.has_prefit = True

    def fit(self, lc_1: lightcurve, lc_2: lightcurve, seed: int = None):
        if seed is None: seed = self.seed
        if not self.is_ready: self.readyup()
        if not self.has_prefit: self.prefit(lc_1, lc_2, seed=seed)

        self.msg_run("Running sampler with %i chains and %i samples" % (self.num_chains, self.num_samples))

        self.sampler.run(jax.random.PRNGKey(seed), self.stat_model.lc_to_data(lc_1, lc_2))
        self.has_run = True

    def get_samples(self, N: int = None, seed: int = None, importance_sampling: bool = False) -> {str: [float]}:
        if seed is None: seed = self.seed
        if not self.has_run:
            self.msg_err("Can't get samples before running!")
        if importance_sampling:
            self.msg_err("JAVELIKE Already distributed according to posterior (ideally)")
        samps = self.sampler.get_samples()

        if not (N is None):
            M = _utils.dict_dim(samps)[1]
            if M > N: self.msg_err("Tried to get %i sub-samples from chain of %i total samples." % (M, N))

            I = np.random.choice(np.arange(M), N, replace=True)
            samps = {key: samps[key][I] for key in samps.keys()}
        return (samps)


# =====================================================
if __name__ == "__main__":
    import mocks

    import matplotlib.pyplot as plt
    from models import dummy_statmodel

    #::::::::::::::::::::

    mock = mocks.mock(seed=2, lag=100, season=0)
    mock01 = mock.lc_1
    mock02 = mock.lc_2
    lag_true = mock.lag

    mock().plot()
    #::::::::::::::::::::
    test_statmodel = dummy_statmodel()

    # ICCF Test
    print("Doing ICCF Testing:")
    test_ICCF = ICCF(Nboot=128, Nterp=1024, Nlags=1024, stat_model=test_statmodel, debug=True)
    print("\tDoing Fit")
    test_ICCF.fit(mock01, mock02)
    print("\tFit done")

    ICCF_samples = test_ICCF.get_samples()['lag']
    print("\tICCF Results , dT = %.2f +/- %.2f compared to true lag of %.2f"
          % (ICCF_samples.mean(), ICCF_samples.std(), lag_true)
          )

    plt.figure()
    plt.hist(ICCF_samples, histtype='step', bins=24)
    plt.axvline(lag_true, ls='--', c='k', label="True Lag")
    plt.axvline(ICCF_samples.mean(), ls='--', c='r', label="Mean Lag")
    plt.axvline(ICCF_samples.mean() + ICCF_samples.std(), ls=':', c='r')
    plt.axvline(ICCF_samples.mean() - ICCF_samples.std(), ls=':', c='r')
    plt.title("ICCF Results")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------
    # Prior Sampling

    print("Doing PriorSampling Testing:")
    test_prior_sampler = prior_sampling(stat_model=test_statmodel)
    print("\tDoing Fit")
    test_prior_sampler.fit(lc_1=mock01, lc_2=mock02, seed=0)
    print("\tFit done")

    test_samples = test_prior_sampler.get_samples(512, importance_sampling=True)
    print("\tPriorSampling Results , dT = %.2f +/- %.2f compared to true lag of %.2f"
          % (test_samples['lag'].mean(), test_samples['lag'].std(), test_statmodel.lag_peak)
          )

    plt.figure()
    plt.title("Prior Sampling")
    plt.hist(test_samples['lag'], histtype='step', density=True)
    plt.axvline(test_statmodel.lag_peak, ls='--', c='k')
    plt.axvline(test_samples['lag'].mean(), ls='--', c='r')
    plt.axvline(test_samples['lag'].mean() + test_samples['lag'].std(), ls=':', c='r')
    plt.axvline(test_samples['lag'].mean() - test_samples['lag'].std(), ls=':', c='r')
    plt.ylabel("Posterior Density")
    plt.xlabel("Lag")
    plt.grid()
    plt.tight_layout()
    plt.show()
