'''
A test script for the hessian scan fitting method

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
from jax.experimental.jax2tf.examples.mnist_lib import step_size
from numpyro import distributions as dist
from tinygp import GaussianProcess
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax
from jax.random import PRNGKey
import jax.numpy as jnp

from litmus_rm import models
from litmus_rm.models import _default_config
from litmus_rm.ICCF_working import *
from litmus_rm import _utils
from litmus_rm.models import stats_model, dummy_statmodel, GP_simple
from litmus_rm.fitting_methods import hessian_scan
from litmus_rm.mocks import mock, mock_A, mock_B, mock_C

from chainconsumer import ChainConsumer

# ============================================
# Generate a mock fit

# mymock = mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau=200.0)
# mymock = mock(cadence=[7, 30], E=[0.1, 0.4], season=180, lag=180, tau=200.0)
# mymock = mock_C.copy(E=[.25, .25], cadence=[30, 30], maxtime=360 * 2, season=60)
# mymock = mock(1, cadence=[60, 60], E=[0.25, 0.25], season=0)
mymock = mock_C.copy(tau=np.exp(5.56), lag=106, cadence=(7, 7), E=[0.1, 0.1], season=0.0)

mymock(12)
f = mymock.plot(true_args={'alpha': 0.0}, show=False)
f.suptitle("True lag = %.2f" % mymock.lag)
plt.show()

# --------------------------------
# Switch for GP simple or dummy model

model = GP_simple()
model.debug = True
data = model.lc_to_data(mymock.lc_1, mymock.lc_2)

# --------------------------------
seed_params = model.find_seed(data)[0]
tol_seed = model.opt_tol(seed_params, data)

print("Initial tolerance: ~ +/- %.2e σ from optimum" % tol_seed)


estmap = model.scan(seed_params, data,
                    precondition="None",
                    optim_kwargs={}
                    )

tol_map = model.opt_tol(estmap,data)
print("Final tolerance: ~ +/- %.2e σ from optimum" % tol_map)


estmap = model.scan(seed_params, data,
                    precondition="half-eig",
                    optim_kwargs={}
                    )

tol_map = model.opt_tol(estmap,data)
print("Final tolerance: ~ +/- %.2e σ from optimum" % tol_map)
