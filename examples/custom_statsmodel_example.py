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
import litmus.gp_working as gpw


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

    def model_function(self, data) -> None:
        lag, logtau, logamp, rel_amp, mean, rel_mean, e_calib = self.prior()

        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag = jnp.square(E)

        delays = jnp.array([0, lag])
        amps = jnp.array([amp, rel_amp * amp])
        means = jnp.array([mean, mean + rel_mean])

        T_delayed = T - delays[bands]
        I = T_delayed.argsort()

        # Build and sample GP

        gp = gpw.build_gp(T_delayed[I], Y[I], diag[I], bands[I], tau, amps, means, basekernel=self.basekernel)
        numpyro.sample("Y", gp.numpyro_dist(), obs=Y[I])
