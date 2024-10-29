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
from numpyro import distributions as dist
from tinygp import GaussianProcess
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax
from jax.random import PRNGKey
import jax.numpy as jnp

from litmus import models
from litmus.models import _default_config
from litmus.ICCF_working import *
from litmus import _utils
from litmus.models import stats_model, dummy_statmodel, GP_simple
from litmus.fitting_methods import hessian_scan
from litmus.mocks import mock, mock_A, mock_B, mock_C

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

'''
test_model.set_priors({'logtau': [np.log(mymock.tau / 10), np.log(mymock.tau * 10)],
                       'logamp': [np.log(0.1), np.log(10.0)],
                       'mean': [-5, 5],
                       'rel_amp': [0.1, 10],
                       'rel_mean': [-5, 5]
                       })
'''

test_model.set_priors({'lag': [0, 1000]})

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------

Nlags = (mymock.maxtime / np.array(mymock.cadence)).max() * 2
Nlags = int(Nlags)
Nlags = 128
Nlags = 32

print("Doing Hessian Fitting with grid of %i lags" % Nlags)
fitting_method = hessian_scan(stat_model=test_model,
                              Nlags=Nlags,
                              init_samples=1_000,
                              grid_bunching=0.8,
                              optimizer_args={'tol': 1E-2,
                                              'maxiter': 256,
                                              'increase_factor': 1.8},
                              reverse=False
                              )

print("Doing prefit in main")
fitting_method.prefit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Doing fit in main")
fitting_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(fitting_method.get_evidence())

# -----------------
# Plotting

fitting_method.diagnostics(plot=True)

f, (a1, a2) = plt.subplots(2, 1, sharex=True)

a1.scatter(fitting_method.scan_peaks['lag'], np.exp(fitting_method.log_evidences),
           label='hessian')
a1.plot(fitting_method.scan_peaks['lag'], np.exp(fitting_method.log_evidences),
        label='hessian')

a2.scatter(fitting_method.scan_peaks['lag'], fitting_method.log_evidences, label='hessian', s=4)
a2.legend()
a2.set_ylim(fitting_method.log_evidences.min() - 100, fitting_method.log_evidences.max())

f.supylabel("Marginal Likelihood")
f.supxlabel("Lag (Days)")

for a in (a1, a2):
    a.axvline(mymock.lag, c='k', ls='--')
    a.grid()

    t = 180
    xlims = plt.gca().get_xlim()
    plt.gca().set_xlim(*xlims)
    while t < xlims[-1]:
        a.axvspan(t, t + 180, alpha=0.15, color='red', zorder=-1)
        t += 360

plt.show()
