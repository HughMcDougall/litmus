import jax
import numpy as np

import litmus.models
from litmus import *
import jax.numpy as jnp
import numpyro


class model_twolag(litmus.models.GP_simple):
    def __init__(self, **kwargs):
        self._default_prior_ranges = {
            'lag_two': [-1000, 0.0],
            'lag_relamp': [0.0, 1.0],
        }
        super().__init__(**kwargs)

    def prior(self) -> list[float, float, float, float, float, float, float, float]:
        lag, logtau, logamp, rel_amp, mean, rel_mean = super().prior()
        lag_two = litmus.models.quickprior(self, 'lag_two')
        lag_relamp = litmus.models.quickprior(self, 'lag_relamp')

        return lag, logtau, logamp, rel_amp, mean, rel_mean, lag_two, lag_relamp

    def model_function(self, data) -> None:
        lag, logtau, logamp, rel_amp, mean, rel_mean, lag_two, lag_relamp = self.prior()

        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag = jnp.square(E)

        means = jnp.array([mean, mean + rel_mean])

        params = {'lag': lag,
                  'logtau': logtau,
                  'logamp': logamp,
                  'rel_amp': rel_amp,
                  'mean': mean,
                  'rel_mean': rel_mean,
                  'lag_two': lag_two,
                  'lag_relamp': lag_relamp,
                  }

        covariance_matrix = self.make_matrix(params, data)
        covariance_matrix += jnp.diag(diag)

        dist = numpyro.distributions.MultivariateNormal(loc=means[bands], covariance_matrix=covariance_matrix)
        numpyro.sample('obs', dist, obs=Y)

        # -------------------------------------------

    def make_matrix_dep(self, params, data):
        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]
        lag = params['lag']
        lag_two = params['lag_two']
        amp1 = jnp.exp(params['logamp'])
        amp2 = jnp.exp(params['logamp']) * params['rel_amp']
        tau = jnp.exp(params['logtau'])

        r = params['lag_relamp']

        index = (2 ** bands[None]).T @ (3 ** bands[None])

        sigs = amp1 * (1 - bands) + amp2 * (bands)
        sigs = sigs[None] * sigs[None].T
        dT = T[None] - T[None].T
        print(dT.shape)

        I_cc = jnp.argwhere(index == 1)
        I_cr = jnp.argwhere(index == 2)
        I_rc = jnp.argwhere(index == 3)
        I_rr = jnp.argwhere(index == 6)

        out = jnp.zeros_like(dT)
        out_cross = out.copy()
        kf = lambda X: jnp.exp(-abs(X / tau))

        dT_cc = dT[I_cc[:, 0], I_cc[:, 1]]
        covar_cc = kf(dT_cc)
        out = out.at[I_cc[:, 0], I_cc[:, 1]].set(covar_cc)
        print('ccs done')

        dT_rr = dT[I_rr[:, 0], I_rr[:, 1]]
        covar_rr = kf(dT_rr)
        covar_rr += kf(dT_rr - (lag + lag_two)) * r * (1 - r)
        covar_rr += kf(dT_rr + (lag + lag_two)) * r * (1 - r)
        out = out.at[I_rr[:, 0], I_rr[:, 1]].set(covar_rr)
        print('rrs done')

        # If incorrect change I_cr to I_rc
        dT_cr = dT[I_cr[:, 0], I_cr[:, 1]]
        covar_cr = kf(dT_cr - lag) * (1 - r)
        covar_cr += kf(dT_cr - lag_two) * r
        out_cross = out_cross.at[I_cr[:, 0], I_cr[:, 1]].set(covar_cr)
        out_cross += out_cross.T

        out += out_cross

        print('rcs done')

        out *= sigs

        return out

    def make_matrix(self, params, data):
        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]
        lag = params['lag']
        lag_two = params['lag_two']
        amp1 = jnp.exp(params['logamp'])
        amp2 = jnp.exp(params['logamp']) * params['rel_amp']
        tau = jnp.exp(params['logtau'])

        r = params['lag_relamp']

        index = (2 ** bands[None]).T @ (3 ** bands[None])
        dT = T[None] - T[None].T

        # todo - make sure kf_cr and kf_rc are the right way around
        lag_12 = lag + lag_two
        kf_cc = lambda X: amp1 ** 2 * jnp.exp(-abs(X / tau))
        kf_cr = lambda X: amp1 * amp2 * (
                jnp.exp(-abs((X - lag) / tau)) * (1 - r) + jnp.exp(-abs((X - lag_two) / tau)) * r)
        kf_rc = lambda X: amp1 * amp2 * (
                jnp.exp(-abs((X + lag) / tau)) * (1 - r) + jnp.exp(-abs((X + lag_two) / tau)) * r)
        kf_rr = lambda X: amp2 ** 2 * (
                jnp.exp(-abs(X / tau)) +
                jnp.exp(-abs((X - lag_12) / tau)) * r * (1 - r) +
                jnp.exp(-abs((X + lag_12) / tau)) * r * (1 - r)
        )

        def f(dT, ind):
            out = jnp.select(
                [
                    ind == 1,
                    ind == 3,
                    ind == 2,
                    ind == 6
                ],
                [
                    kf_cc(dT),
                    kf_cr(dT),
                    kf_rc(dT),
                    kf_rr(dT)
                ],
                default=0.0
            )
            return out

        f = jax.vmap(f, in_axes=(0, 0), out_axes=(0))
        f = jax.vmap(f, in_axes=(1, 1), out_axes=(1))

        out = f(dT, index)

        out = 0.5 * (out + out.T)

        return out
