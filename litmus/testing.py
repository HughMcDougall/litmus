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

import models
from models import _default_config

from functools import partial

from ICCF_working import *

from models import stats_model
from chainconsumer import ChainConsumer

if __name__ == "__main__":
    from mocks import mock_A_01, mock_A_02, lag_A
    from mocks import mock_B_01, mock_B_02, lag_B
    from mocks import mock_C_01, mock_C_02, lag_C

    import matplotlib.pyplot as plt

    T = jnp.array([*mock_C_01.T, *mock_C_02.T])
    Y = jnp.array([*mock_C_01.Y, *mock_C_02.Y])
    E = jnp.array([*mock_C_01.E, *mock_C_02.E])
    bands = jnp.array([*np.zeros(mock_C_01.N), *np.ones(mock_C_02.N)]).astype(int)

    I = T.argsort()[::2]

    T, Y, E, bands = T[I], Y[I], E[I], bands[I]

    test_data = {'T': T,
                 'Y': Y,
                 'E': E,
                 'bands': bands
                 }

    # ---------------------
    true_params = {'lag': lag_C,
                   'logamp': 0,
                   'logtau': np.log(400),
                   'mean': 0,
                   'rel_mean': 0,
                   'rel_amp': 1
                   }

    test_statmodel = models.GP_simple()

    test_statmodel.set_priors({
        key: [(true_params[key]+0.01)*0.9,(true_params[key]+0.01)*1.1]
        for key in true_params.keys()
    })

    print("Generating samples")

    test_params = test_statmodel.prior_sample(num_samples=10_000, data=test_data)

    '''
    for key in test_params.keys():
        if key != 'lag':
            test_params[key] = test_params[key]*0 + true_params[key]
    '''

    print("Getting log-likelihoods")

    log_likes = test_statmodel.log_likelihood(data=test_data, params=test_params)

    weights = np.exp(log_likes - log_likes.max())
    weights /= weights.sum()


    C = ChainConsumer()
    C.add_chain(test_params, weights=weights)
    C.plotter.plot(truth=true_params, extents = test_statmodel.prior_ranges)
    plt.show()
    # ---------------------
