'''
Contains NumPyro generative models.

HM 24
'''

# ============================================
# IMPORTS
import sys

import jax.scipy.optimize
import numpyro
from numpyro.distributions import MixtureGeneral
from numpyro import distributions as dist
from numpyro import handlers

from tinygp import GaussianProcess
from jax.random import PRNGKey
import jax.numpy as jnp
import tinygp
from gp_working import *

import scipy

from _utils import *
import jaxopt

# ============================================
#

# TODO - update these
_default_config = {
    'logtau': (0, 10),
    'logamp': (0, 10),
    'rel_amp': (0, 10),
    'mean': (-50, 50),
    'rel_mean': 1.0,
    'lag': (0, 1000),

    'outlier_spread': 10.0,
    'outlier_frac': 0.25,
}


# ============================================
# Base Class

class stats_model(object):
    '''
    Base class for bayesian generative models. Includes a series of utilities for evaluating likelihoods, gradients etc,
    as well as various

    Todo:
    - Change prior volume calc to be a function call for flexibility
    - Add kwarg support to model_function and model calls to be more flexible / allow for different prior types
    - Fix the _scan method to use jaxopt and be jitted / vmapped
    - Add Hessian & Grad functions
    '''

    def __init__(self, prior_ranges=None):

        if not hasattr(self, "_default_prior_ranges"):
            self._default_prior_ranges = {
                'lag': _default_config['lag'],
            }
        self.prior_ranges = {} | self._default_prior_ranges  # Create empty priors
        self.prior_volume = 1.0

        # Update with args
        self.set_priors(self._default_prior_ranges | prior_ranges) if prior_ranges is not None else self.set_priors(
            self._default_prior_ranges)

        self._log_likelihood_vector = lambda data, params: self._log_likelihood(data=data, params=params)
        self._log_likelihood_vector = jax.vmap(self._log_likelihood_vector,
                                               in_axes=(None,
                                                        {key: 0 for key in self._default_prior_ranges.keys()}
                                                        )
                                               )

        self._log_likelihood_vector = jax.jit(self._log_likelihood_vector)
        self._log_likelihood_single = jax.jit(self._log_likelihood)
        self._log_likelihood_single.__doc__ = "Jitted single eval likelihood"
        self._log_likelihood_vector.__doc__ = "Jitted array eval likelihood"

        return

    def set_priors(self, prior_ranges):
        '''
        Sets the stats model prior ranges for uniform priors. Does some sanity checking to avoid negative priors
        :param prior_ranges:
        :return: 
        '''

        prior_volume = 1.0

        badkeys = [key for key in prior_ranges.keys() if key not in self._default_prior_ranges.keys()]

        for key, val in zip(prior_ranges.keys(), prior_ranges.values()):
            if key in badkeys: continue

            assert (len(val) == 2), "Bad input shape in set_priors for key %s" % key  # todo - make this go to std.err
            a, b = float(min(val)), float(max(val))
            self.prior_ranges[key] = [a, b]
            prior_volume *= b - a

            print(key, a, b, prior_volume)

        self.prior_volume = prior_volume

        return

    # --------------------------------
    def model_function(self, data):
        '''
        A NumPyro callable function
        '''
        lag = numpyro.sample('lag', dist.Uniform(self.prior_ranges['lag'][0], self.prior_ranges['lag'][1]))

        numpyro.sample('test_sample', dist.Normal(lag, 100), obs=data[0])

    # --------------------------------
    # Un-Jitted Functions

    def _log_likelihood(self, data, params):
        '''
        Raw, un-jitted and un-vmapped log likelihood evaluation
        '''
        out = numpyro.infer.util.log_density(self.model_function, (), {'data': data}, params)[0]
        return (out)

    def _grad(self, data, params):
        '''
        Un-jitted gradient evaluation
        '''

    def _hessian(self):
        '''
        Un-jitted hessian evaluation
        '''
        return()

    def _scan(self, fixed_params, init_params=None, data=None, tol=1E-3):
        '''
        un-jitted optimizer over fixed and free vars
        :param fixed_params: dict
        :param init_params: dict
        :param data: model arguments
        :return:
        '''

        # If init_params is empty or incomplete, fill in with values drawn from prior
        if init_params is None: init_params = {}
        if (fixed_params | init_params).keys() != self._default_prior_ranges.keys():
            prior_params = self.prior_sample(data=data, num_samples=1)
            prior_params |= init_params
            init_params = {key: prior_params[key] for key in prior_params.keys() if key not in fixed_params.keys()}
        print(init_params)

        # array-like function for use with optimize
        x0 = jnp.array(
            [float(val) for val in list(init_params.values())]
        )

        def f(x):
            free_params = {key: val for key, val in zip(init_params.keys(), x)}
            params = fixed_params | free_params
            out = -1.0 * self._log_likelihood(data=data, params=params)
            print(params, out)
            return (out)

        res = scipy.optimize.minimize(fun=f, x0=x0, method='Nelder-Mead', tol=1E-8)

        return (res.x)

    # --------------------------------
    # Wrapped Functions

    def log_likelihood(self, data, params):
        '''
        Gives the log likelihood of a set of sample params
        '''
        print(params.keys(), self._default_prior_ranges.keys())
        assert params.keys() == self._default_prior_ranges.keys(), "Tried to call log_likelihood with bad parameter names"

        if not isiter_dict(params):
            out = self._log_likelihood_single(data, params)
        else:
            out = self._log_likelihood_vector(data, params)

        return out

    # --------------------------------
    # Utils
    def prior_sample(self, data=None, num_samples=1):
        '''
        Blind sampling from the prior without conditioning
        '''

        pred = numpyro.infer.Predictive(self.model_function,
                                        num_samples=num_samples,
                                        return_sites=list(self._default_prior_ranges.keys())
                                        )

        params = pred(rng_key=jax.random.PRNGKey(np.random.randint(0, sys.maxsize // 1024)), data=data)
        return (params)

# ============================================
# Custom statmodel example
class dummy_statmodel(stats_model):
    '''
    An example of how to construct your own stats_model in the simplest form.
    Requirements are to:
        1. Set a default prior range for all parameters used in model_function
        2. Define a numpyro generative model model_function
    You can add / adjust methods as required, but these are the only main steps
    '''

    def __init__(self, prior_ranges=None):
        self._default_prior_ranges = {
            'lag': _default_config['lag'],
            'test_param': [0.0, 1.0]
        }
        super().__init__(prior_ranges=prior_ranges)

    def model_function(self, data):
        lag = numpyro.sample('lag', dist.Uniform(self.prior_ranges['lag'][0], self.prior_ranges['lag'][1]))
        test_param = numpyro.sample('test_param', dist.Uniform(self.prior_ranges['test_param'][0],
                                                               self.prior_ranges['test_param'][1]))
        numpyro.sample('test_sample', dist.Normal(lag, 100), obs=data[0])
        numpyro.sample('test_sample_2', dist.Normal(test_param, 1.0), obs=data[1])

# ============================================
# ============================================
# Testing

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Build a test stats model and adjust priors
    test_statmodel = dummy_statmodel()
    test_statmodel.set_priors({
        'lag': [0, 500],
        'alpha': [0, 1]
    })

    # Generate prior samples and evaluate likelihoods
    test_data = jnp.array([100., 0.25])
    test_params = test_statmodel.prior_sample(num_samples=1_000, data=test_data)
    log_likes = test_statmodel.log_likelihood(data=test_data, params=test_params)

    # Try a scan
    opt_lag = test_statmodel._scan(fixed_params={}, data=test_data, init_params={'lag': 300.0})
    print("best lag is", opt_lag)
    # ------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
    a1.scatter(test_params['lag'], np.exp(log_likes))
    a2.scatter(test_params['lag'], log_likes)

    for a in (a1, a2): a.grid()
    fig.supxlabel("Lag (days)")
    a1.set_ylabel("Un-Normalized Posterior")
    a2.set_ylabel("Un-Normalized Log-Posterior")
    fig.tight_layout()

    plt.show()
