'''
This is an example of how to implement your own stats model
'''

# ============================================
from litmus.models import _default_config, stats_model, quickprior
import numpyro
import numpyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

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
            'logtau': _default_config['logtau'],
            'logamp': _default_config['logamp']
        }
        super().__init__(prior_ranges=prior_ranges)
        self.lag_peak = 250.0
        self.tau_peak = 0.5

        self.evidence_approx = np.product(np.array([np.ptp(x) for x in self.prior_ranges.values()])) ** -1

    # ----------------------------------
    def prior(self):

        lag = quickprior(self, 'lag')
        logtau = quickprior(self, 'logtau')
        logamp = quickprior(self, 'logamp')

        return (lag, logtau, logamp)

    def model_function(self, data):
        lag, logtau, logamp = self.prior()

        numpyro.sample('test_sample', dist.Normal(lag, 25), obs=self.lag_peak)
        numpyro.sample('test_sample_2',
                       dist.MultivariateNormal(
                           loc=jnp.array([0.25, 0.5]),
                           covariance_matrix=jnp.array([
                               [0.25, 0.00],
                               [0.00, 0.25]
                           ])),
                       obs=jnp.array([logtau, logamp])
                       )


