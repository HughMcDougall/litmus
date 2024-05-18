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


# ============================================

# Base fitting procedure class
class fitting_procedure(object):
    '''
    Generic class for lag fitting procedures. Contains parent methods for setting properties
    '''

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        if not hasattr(self, "_default_params"):
            self._default_params = {}

        self.debug = debug
        self.out_stream = out_stream
        self.err_stream = err_stream

        self.is_ready = False
        self.has_run = False

        self.fitting_params = {} | self._default_params
        self.set_config(**(self._default_params | fit_params))

        self.results = None

        self.seed = np.random.randint(0, 2 ** 16) if "seed" not in fit_params.keys() else fit_params['seed']

    def fit(self, lc_1, lc_2, seed=None, stat_model=None):
        '''
        :param lc_1:
        :param lc_2:
        :param seed:
        :return:
        '''
        seed = seed if isinstance(seed, int) else self.seed
        print("This fitting method does not have method .fit()", file=self.err_stream)

        return

    def reset(self):
        '''
        Clears all memory and resets params to defaults
        '''
        self.set_config(**self._default_params)

        self.has_run, self.is_ready = False, False
        self.results = None

        return

    def get_samples(self, N=None):
        print("This fitting method does not have method .get_samples()", file=self.err_stream)

    def set_config(self, **fit_params):
        '''
        Configure fitting parameters for fitting_method() object
        Accepts any parameters present with a name in fitting_method.fitting_params
        Unlisted parameters will be ignored
        '''

        if self.debug: print("Doing config with keys", fit_params.keys())

        badkeys = [key for key in fit_params.keys() if key not in self._default_params.keys()]

        for key, val in zip(fit_params.keys(), fit_params.values()):
            if key in badkeys: continue

            # If something's changed, flag as having not run
            if self.has_run and val != self.__getattr__(key): self.has_run = False

            self.__setattr__(key, val)
            self.fitting_params |= {key: val}
            if self.debug: print("\t set attr", key, file=self.out_stream)

        if len(badkeys) > 0:
            print("Tried to configure bad keys:", end="\t", file=self.err_stream)
            for key in badkeys: print(key, end=', ', file=self.err_stream)

        return


# ============================================
# ICCF fitting procedure

class ICCF(fitting_procedure):
    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        self._default_params = {
            'Nboot': 512,
            'Nterp': 2014,
            'lag_range': _default_config['lag'],
        }

        super().__init__(out_stream=out_stream, err_stream=err_stream, debug=debug, **fit_params)

        self.results = {'samples': np.zeros(self.Nboot),
                        'correl_curve': np.zeros(self.Nterp),
                        'lag_mean': 0.0,
                        'lag_err': 0.0
                        }

    def fit(self, lc_1, lc_2, seed=None, stat_model=None):
        # Unpack lightcurve
        X1, Y1, E1 = lc_1.T, lc_1.Y, lc_1.E
        X2, Y2, E2 = lc_2.T, lc_2.Y, lc_2.E

        if seed is None:
            seed = self.seed

        # Do bootstrap fitting
        jax_samples = correl_func_boot_jax_wrapper_nomap(self.lags, X1, Y1, X2, Y2, E1, E2, Nterp=self.Nterp,
                                                         Nboot=self.Nboot)

        self.results['samples'] = jax_samples
        self.results['lag_mean'] = jax_samples.mean()
        self.results['lag_err'] = jax_samples.std()
        self.has_run = True

    def get_samples(self, N=None):
        if N is None:
            return (self.results['samples'])
        else:
            if N > self.Nboot:
                print("Warning, tried to get %i sub-samples from %i boot-strap itterations in ICCF" % (N, self.Nboot),
                      file=self.err_stream)
            return (np.random.choice(self.results['samples'], N, replace=True))

    def set_config(self, **fit_params):
        super().set_config(**fit_params)
        self.lags = np.linspace(self.lag_range[0], self.lag_range[1], self.Nterp)


# ============================================
# Random Prior Sampling

class prior_sampling(fitting_procedure):
    '''
    Randomly samples from the prior and weights with importance sampling
    todo - change the way lag_range is generated to call a stats model
    '''

    def __init__(self, out_stream=sys.stdout, err_stream=sys.stderr, debug=False, **fit_params):

        self._default_params = {
            'Nsamples': 4096
        }

        super().__init__(out_stream=out_stream, err_stream=err_stream, debug=debug, **fit_params)

        self.results = {'samples': np.zeros(self.Nsamples),
                        'weights': np.zeros(self.Nsamples)}

    def get_samples(self, N=None):
        if N is None:
            return (self.results['samples'])
        else:
            if N > self.Nsamples:
                print("Warning, tried to get %i sub-samples from %i boot-strap itterations in ICCF" % (N, self.Nboot),
                      file=self.err_stream)
            return (np.random.choice(self.results['samples'], N, replace=True))


if __name__ == "__main__":
    from mocks import data_1, data_2

    #::::::::::::::::::::
    # Make Litmus Object
    test_ICCF = ICCF(Nboot=512, Nterp=1024)
    test_ICCF.fit(data_1, data_2)
    samples = test_ICCF.get_samples(1024)
    print(samples.mean(), samples.std())
