'''
A test script for the Nested Sampling Fitting Method

HM 24
'''

# ============================================
# IMPORTS
import os, sys
from functools import partial

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('module://backend_interagg')
import matplotlib
import numpy as np

import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from numpyro import infer

import jax
from jax.random import PRNGKey
import jax.numpy as jnp

from litmus_rm import models
from litmus_rm.models import _default_config
from litmus_rm.ICCF_working import *
from litmus_rm import _utils
from litmus_rm.models import stats_model, dummy_statmodel, GP_simple
from litmus_rm.fitting_methods import *
from litmus_rm.mocks import mock, mock_A, mock_B, mock_C

from chainconsumer import ChainConsumer

# ============================================
# Generate a mock fit

mymock = mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau=200.0)
# mymock = mock(cadence=[7, 30], E=[0.1, 0.4], season=180, lag=180, tau=200.0)

mymock(12)
f = mymock.plot(true_args={'alpha': 0.0}, show=False)
f.suptitle("True lag = %.2f" % mymock.lag)
plt.show()

# --------------------------------

# Switch for GP simple or dummy model

test_model = GP_simple()


data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------
from jaxns import NestedSampler, Model, Prior
import tensorflow_probability.substrates.jax as tfp
import jaxns

tfpd = tfp.distributions

bounds = np.array([a[1] for a in test_model.prior_ranges.items()])
lo, hi = jnp.array(bounds[:, 0]), jnp.array(bounds[:, 1])

true_params = mymock.params()


def prior_model():
    x = yield Prior(tfpd.Uniform(low=lo, high=hi), name='x')
    return x


def log_likelihood(x):
    params = _utils.dict_unpack(x, keys = test_model.paramnames())
    with numpyro.handlers.block(hide=test_model.paramnames()):
        LL = test_model._log_likelihood(params, data)
    return LL


model = Model(prior_model=prior_model,
              log_likelihood=log_likelihood,
              )

print("Making nested sampler")
ns = NestedSampler(model=model,
                   max_samples=100_000,
                   verbose=False,
                   num_live_points=2_000,
                   num_parallel_workers=1,
                   difficult_model=True,
                   )
term = jaxns.TerminationCondition(dlogZ=np.log(1+np.log(1E-4)))

print("Running nested sampler")
termination_reason, state = ns(jax.random.PRNGKey(42), term)
results = ns.to_results(termination_reason=termination_reason, state=state)
ns.plot_diagnostics(results)
ns.plot_cornerplot(results)
