'''
A quick and nasty test file to confirm everything's working

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

import models
from models import _default_config

from functools import partial

from ICCF_working import *

import _utils

from models import stats_model
from chainconsumer import ChainConsumer

# ============================================

if __name__ == "__main__":
    from mocks import mock_A_01, mock_A_02, lag_A
    from mocks import mock_B_01, mock_B_02, lag_B
    from mocks import mock_C_01, mock_C_02, lag_C

    import matplotlib.pyplot as plt

    T = jnp.array([*mock_A_01.T, *mock_A_02.T])
    Y = jnp.array([*mock_A_01.Y, *mock_A_02.Y])
    E = jnp.array([*mock_A_01.E, *mock_A_02.E])
    bands = jnp.array([*np.zeros(mock_A_01.N), *np.ones(mock_A_02.N)]).astype(int)

    I = T.argsort()

    T, Y, E, bands = T[I], Y[I], E[I], bands[I]

    test_data = {'T': T,
                 'Y': Y,
                 'E': E,
                 'bands': bands
                 }

    # ---------------------
    true_params = {'lag': lag_C,
                   'logamp': 0.0,
                   'logtau': np.log(1000),
                   'mean': 0.0,
                   'rel_mean': 0.0,
                   'rel_amp': 1.0
                   }

    test_statmodel = models.GP_simple()

    # Fix most params as deltas
    test_statmodel.set_priors({
        key: [true_params[key], true_params[key]]
        for key in true_params.keys() if key not in ['lag', 'logamp']
    })

    # Make tight but non-trival ranges for remainders
    test_statmodel.set_priors({'lag': [0.0, 200.0],
                               'logtau': [6, 8],}
                              )

    plt.figure()
    plt.errorbar(T - np.array([0, lag_C])[bands], Y, E, fmt='none')
    plt.show()

    print("Generating samples")
    test_params = test_statmodel.prior_sample(num_samples=10_000, data=test_data)

    if True:
        print("Samples generating. Getting likelihoods.")
        log_likes_samples = test_statmodel.log_likelihood(data=test_data, params=test_params, use_vmap=False)

        print("Done! Plotting now")

        weights = np.exp(log_likes_samples - log_likes_samples.max())
        weights /= weights.sum()

        C = ChainConsumer()
        C.add_chain({key: test_params[key] for key in ['lag', 'logtau']}, weights=weights, cloud=True)
        C.plotter.plot(truth=true_params, extents=test_statmodel.prior_ranges)
        plt.show()

        plt.figure()
        I = np.where(abs(test_params['logtau'] - true_params['logtau']) < 0.05)
        #plt.plot(test_params['lag'][I], log_likes_samples[I])
        plt.scatter(test_params['lag'][I], log_likes_samples[I])
        plt.axvline(true_params['lag'])
        plt.grid()
        plt.show()

    if True:
        lags = np.linspace(0, 200, 1024)
        log_likes = test_statmodel.log_likelihood(data=test_data, params=_utils.dict_extend(true_params, {'lag': lags}))

        finite_grads = np.diff(log_likes) / np.diff(lags)
        grads = np.array([test_statmodel.grad(data=test_data, params=true_params | {'lag': lag})[0] for lag in lags])

        hess = np.array(
            [test_statmodel.hessian(data=test_data, params=true_params | {'lag': lag})[0][0] for lag in lags])
        finite_hess = np.diff(grads) / np.diff(lags)

        fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(6, 12), sharex=True)

        a1.plot(lags, log_likes, label='log likelihood')

        a2.plot(lags, np.exp(log_likes - log_likes.max()), label='Likelihood')

        a3.plot((lags[1:] + lags[:-1]) / 2, finite_grads, label='finite diff grads')
        a3.plot(lags, grads, label='Autograd grads')

        a4.scatter((lags[1:] + lags[:-1]) / 2, finite_hess, label='fin=ite diff Hessians')
        a4.plot(lags, hess, label='Autograd Hessians', color='tab:orange')

        for a in (a1, a2, a3, a4):
            a.grid()
            a.legend()
            a.set_xlim(lags.min(), lags.max())
            a.axvline(lag_C)
        # a4.set_ylim(-0.1,0.1)

        fig.tight_layout()
        plt.show()
        # ---------------------
