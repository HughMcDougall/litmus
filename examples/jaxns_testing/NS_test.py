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

import litmus
from litmus import _utils
from litmus.models import stats_model, dummy_statmodel, GP_simple
from litmus.fitting_methods import *
from litmus.mocks import mock, mock_A, mock_B, mock_C

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

test_model.set_priors({key: [mymock.params()[key], mymock.params()[key]] for key in mymock.params().keys()})
test_model.set_priors({'lag': [0, 1000]})
test_model.set_priors({'logtau': [0, 10]})

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------


print("Doing Nested Sampling Fitting")
fitting_method = nested_sampling(stat_model=test_model,
                                 max_samples=1_000,
                                 num_live_points=150,
                                 )

print("Doing prefit in main")
fitting_method.prefit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Doing fit in main")
fitting_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(fitting_method.get_evidence())

# -----------------
# Plotting

fitting_method.diagnostics()
handler = litmus.LITMUS(fitting_method)
handler.lag_plot()
handler.plot_parameters(prior_extents=False)
handler.diagnostic_plots()

fitting_method.sampler.plot_diagnostics(fitting_method._jaxnsresults)
fitting_method.sampler.plot_cornerplot(fitting_method._jaxnsresults)