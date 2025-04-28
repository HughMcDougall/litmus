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
from litmus.models import stats_model, GP_simple
from litmus.fitting_methods import hessian_scan, SVI_scan
from litmus.mocks import mock, mock_A, mock_B, mock_C

from chainconsumer import ChainConsumer

# ============================================
# Generate a mock fit

# mymock = mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau=200.0)
# mymock = mock(cadence=[7, 30], E=[0.1, 0.4], season=180, lag=180, tau=200.0)
# mymock = mock_C.copy(E=[.25, .25], cadence=[30, 30], maxtime=360 * 2, season=60)
mymock = mock(1, cadence=[60, 60], E=[0.25, 0.25], season=0)

mymock(12)
f = mymock.plot(true_args={'alpha': 0.0}, show=False)
f.suptitle("True lag = %.2f" % mymock.lag)
plt.show()

# --------------------------------

# Switch for GP simple or dummy model

test_model = GP_simple()

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------

Nlags = (mymock.maxtime / np.array(mymock.cadence)).max() * 2
Nlags = int(Nlags)
Nlags = 32

print("Doing Hessian Fitting with grid of %i lags" % Nlags)
fitting_method = hessian_scan(stat_model=test_model,
                              Nlags=Nlags,
                              init_samples=5_000,
                              grid_bunching=0.5,
                              optimizer_args={'tol': 1E-4,
                                              'increase_factor': 1.1, },
                              optimizer_args_init={'tol': 1E-10,
                                                   'maxiter': 1024,
                                                   },
                              reverse=False,
                              verbose=2,
                              debug=False,
                              precondition="diag"
                              )

print("Doing prefit in main")
fitting_method.prefit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Doing fit in main")
fitting_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(fitting_method.get_evidence())

# -----------------
# Plotting

fitting_method.diagnostics(show=True)
# fitting_method.refit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
fitting_method.diagnostics(show=True)

# -----------------
# Plotting
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
