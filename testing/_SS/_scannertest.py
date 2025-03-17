'''
A test file to ensure the stats_model._scanner and stats_model.scana are working properly
'''


# ============================================
# IMPORTS
import os, sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax
from jax.random import PRNGKey
import jax.numpy as jnp

from litmus import models
from litmus.models import _default_config
from litmus.ICCF_working import *
from litmus import _utils
from litmus.models import stats_model, dummy_statmodel, GP_simple
from litmus.fitting_methods import fitting_procedure, nested_sampling, prior_sampling, hessian_scan
from litmus.mocks import mock, mock_A, mock_B, mock_C

from chainconsumer import ChainConsumer

# ============================================
# Generate a mock fit

mymock = mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau=200.0)
mymock = mock(cadence=[7, 30], E=[0.1, 0.4], season=180, lag=180, tau=200.0)
mymock(12)
mymock.plot(true_args={'alpha': 0.0})
plt.title("True lag = %.2f" % mymock.lag)
plt.show()

# ============================================
# Make a model

test_model = models.GP_simple()
test_model.debug=True
data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)
truth = mymock.params()

start_params = test_model.find_seed(data)[0]|{'lag':truth['lag']}
params_toscan = [x for x in test_model.paramnames() if x!='lag']

print("Initial position")
for x in start_params.items(): print(*x)
print("Init value: %.2f" %test_model.log_density(start_params, data))

# ============================================

optim_kwargs = {'stepsize': 0.0,
                'min_stepsize': 1E-5,
                'increase_factor': 1.5,
                'maxiter': 1024,
                'linesearch': 'backtracking',
                'verbose': False,
                }


print("Doing scan with .scan \n \n")
from_scan = test_model.scan(start_params=start_params, data=data, optim_params = params_toscan,
                            optim_kwargs=optim_kwargs
                            )
print("Final position")
for x in from_scan.items(): print(*x)
print("Final value: %.2f" %test_model.log_density(from_scan, data))


# ============================================
solver, runsolver, [converter, deconverter, optfunc, runsolver_jit] = test_model._scanner(data, optim_params= params_toscan, optim_kwargs=optim_kwargs, return_aux=True)
from_scanner, aux_data = runsolver(solver, start_params, aux = True)


print("Final position")
for x in from_scanner.items(): print(*x)
print("Final value: %.2f" %test_model.log_density(from_scanner, data))

# ============================================
x0,y0 = converter(start_params)
state = solver.init_state(x0, y0, data)
params_fromscannerjit, state_fromscannerjit = runsolver_jit(solver, start_params, state)

print("Final position")
for x in params_fromscannerjit.items(): print(*x)
print("Final value: %.2f" %test_model.log_density(params_fromscannerjit, data))
# ============================================



f, (a1,a2,a3) = plt.subplots(3,1, sharex=True, sharey=True)
mymock.corrected_plot(axis=a1)
mymock.corrected_plot(from_scan, axis=a2)
mymock.corrected_plot(params_fromscannerjit, axis=a3)

plt.show()
