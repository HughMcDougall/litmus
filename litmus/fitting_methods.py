'''
Contains fitting procedures to be executed by the litmus class object

HM 24
'''

# ============================================
# IMPORTS
import sys

import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from jax.random import PRNGKey

import jax
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax.numpy as jnp
import numpy as np

from models import _default_config

from functools import partial

from ICCF_working import *

from models import stats_model


# ============================================

# Base fitting procedure class
class fitting_procedure(object):
    '''
    Generic class for lag fitting procedures. Contains parent methods for setting properties
    '''

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        if not hasattr(self, "_default_params"):
            self._default_params = {}
        if not hasattr(self, "results"):
            self.results = {}

        self.debug = debug
        self.out_stream = out_stream
        self.err_stream = err_stream
        self.name = "Base Fitting Procedure"

        self.is_ready = False
        self.has_run = False

        self.fitting_params = {} | self._default_params

        self.set_config(**(self._default_params | fit_params))

        self.seed = np.random.randint(0, 2 ** 16) if "seed" not in fit_params.keys() else fit_params['seed']

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
        self.results = None

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

        return

    # ----------------------

    def fit(self, lc_1, lc_2, seed=None, stat_model=None):
        '''
        Fit lags
        :param lc_1: Lightcurve 1 (Main)
        :param lc_2: Lightcurve 2 (Response)
        :param stat_model: a statistical model object
        :param seed: A random seed for feeding to the fitting process. If none, will select randomly
        '''
        seed = seed if isinstance(seed, int) else self.seed
        print("Fitting \"%s\" method does not have method .fit() implemented" % self.name, file=self.err_stream)

        return

    def get_samples(self, N=None):
        '''
        Returns MCMC-like posterior samples
        :param N: Number of samples to return. If None, return all

        :return: keyed dictionary of model parameters
        '''
        print("Fitting \"%s\" method does not have method .get_samples() implemented" % self.name, file=self.err_stream)

    def get_evidence(self):
        '''
        :return:
        '''
        print("Fitting \"%s\" method does not have method .get_evidence() implemented" % self.name,
              file=self.err_stream)

        return (0.0)

    def get_pvalue(self):
        '''
        :return:
        '''
        print("Fitting \"%s\" method does not have method .get_pvalue() implemented" % self.name, file=self.err_stream)

        return (0.0)

    def get_FPR(self):
        '''
        :return:
        '''
        print("Fitting \"%s\" method does not have method .get_FPR() implemented" % self.name, file=self.err_stream)

        return (0.0)


# ============================================
# ICCF fitting procedure

class ICCF(fitting_procedure):
    '''
    Fit lags using interpolated cross correlation function
    todo
        - Change the way lag_range is generated to call from the stats model prior
        - Add p value, false positive and evidence estimates
        - Likelihood importance sampling (?)
        - Add correlation curve / pearson curve
        - Change __getattr__ and __setattr_ to make anything in fitting_params or results work as an alias
    '''

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        self._default_params = {
            'Nboot': 512,
            'Nterp': 2014,
            'Nlags': 512,
            'lag_range': _default_config['lag'],
        }

        super().__init__(out_stream=out_stream, err_stream=err_stream, debug=debug, **fit_params)

        self.name = "ICCF Fitting Procedure"

        self.results = {'samples': np.zeros(self.Nboot),
                        'correl_curve': np.zeros(self.Nterp),
                        'lag_mean': 0.0,
                        'lag_err': 0.0
                        }

    def fit(self, lc_1, lc_2, stat_model=None, seed=None):
        # Unpack lightcurve
        X1, Y1, E1 = lc_1.T, lc_1.Y, lc_1.E
        X2, Y2, E2 = lc_2.T, lc_2.Y, lc_2.E

        if seed is None:
            seed = self.seed

        # Do bootstrap fitting
        jax_samples = correl_func_boot_jax_wrapper_nomap(self.lags, X1, Y1, X2, Y2, E1, E2,
                                                         Nterp=self.Nterp,
                                                         Nboot=self.Nboot)

        self.results['samples'] = jax_samples
        self.results['lag_mean'] = jax_samples.mean()
        self.results['lag_err'] = jax_samples.std()
        self.has_run = True

    def get_samples(self, N=None, importance_sampling=False, stat_model=None):

        if importance_sampling:
            print("Warning! Cannot use important sampling with ICCF. Try implementing manually")
            return

        if N is None:
            return ({'lag': self.results['samples']})
        else:
            if N > self.Nboot:
                print("Warning, tried to get %i sub-samples from %i boot-strap itterations in ICCF" % (N, self.Nboot),
                      file=self.err_stream)
            return ({'lag': np.random.choice(a=self.results['samples'], size=N, replace=True)})

    def set_config(self, **fit_params):
        super().set_config(**fit_params)
        self.lags = np.linspace(self.lag_range[0], self.lag_range[1], self.Nlags)


# ============================================
# Random Prior Sampling

class prior_sampling(fitting_procedure):
    '''
    Randomly samples from the prior and weights with importance sampling
    '''

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        self._default_params = {
            'Nsamples': 4096
        }

        super().__init__(out_stream=out_stream, err_stream=err_stream, debug=debug, **fit_params)

        self.name = "Prior Sampling Fitting Procedure"

        self.results = {'samples': np.zeros(self.Nsamples),
                        'weights': np.zeros(self.Nsamples)
                        }

    # --------------
    def fit(self, lc_1, lc_2, seed=None, stat_model=None):
        # Generate samples & calculate likelihoods
        samples = stat_model.prior_sample(data=None, num_samples=self.Nsamples)
        log_likes = stat_model.log_likelihood(data=(lc_1, lc_2), params=samples)
        likes = np.exp(log_likes)

        # Store results
        self.results['samples'] = samples
        self.results['weights'] = likes

        # Mark good for retrieval
        self.has_run = True

    def get_samples(self, N=None, importance_sampling=True, stat_model=None):

        if N is None:
            N = self.Nsamples
        else:
            if N > self.Nsamples:
                print("Warning, tried to get %i sub-samples from %i samples" % (N, self.Nsamples),
                      file=self.err_stream)

        if importance_sampling:
            weights = self.results['weights'] / self.results['weights'].sum()
        else:
            weights = None

        I = np.random.choice(a=np.arange(self.Nsamples), size=N, replace=True,
                             p=weights)
        return ({
            key: val[I] for key, val in zip(self.results['samples'].keys(), self.results['samples'].values())
        })


if __name__ == "__main__":
    from mocks import mock_A_01, mock_A_02, lag_A
    from mocks import mock_B_01, mock_B_02, lag_B
    from mocks import mock_C_01, mock_C_02, lag_C

    import matplotlib.pyplot as plt
    from models import dummy_statmodel

    #::::::::::::::::::::
    # ICCF Test
    test_ICCF = ICCF(Nboot=1024, Nterp=1024, Nlags=1024)
    print("Doing Fit")
    # test_ICCF.fit(mock_C_01, mock_C_02)
    print("Fit done")

    ICCF_samples = test_ICCF.get_samples()['lag']
    print(ICCF_samples.mean(), ICCF_samples.std())

    plt.figure()
    plt.hist(ICCF_samples, histtype='step', bins=24)
    plt.axvline(lag_C, ls='--', c='k', label="True Lag")
    plt.axvline(ICCF_samples.mean(), ls='--', c='r', label="Mean Lag")
    plt.title("ICCF Results")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------
    test_statmodel = dummy_statmodel()

    test_prior_sampler = prior_sampling()
    test_prior_sampler.fit(lc_1=None, lc_2=None, seed=0, stat_model=test_statmodel)
    test_samples = test_prior_sampler.get_samples(512)

    plt.figure()
    plt.title("Dummy prior sampling test")
    plt.hist(test_samples['lag'], histtype='step')
    plt.axvline(250.0, ls='--', c='k')
    plt.grid()
    plt.show()
