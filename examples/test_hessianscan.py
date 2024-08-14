'''
A test script for the hessian scan fitting method

HM 24
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

mymock = mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau = 200.0)
mymock.plot(true_args={'alpha': 0.0})
plt.show()

# --------------------------------

# Switch for GP simple or dummy model
if True:
    test_model = GP_simple()
    test_model.set_priors({'lag': [0, 500]})
    '''
    test_model.set_priors({'logtau': [np.log(mymock.tau * 0.1), np.log(mymock.tau * 10)],
                           'logamp': [0, 0],
                           'mean': [0, 0],
                           'rel_amp': [1, 1],
                           'rel_mean': [0, 0]
                           })
    '''
    test_model.set_priors({'logtau': [np.log(mymock.tau / 10), np.log(mymock.tau * 10)],
                           'logamp': [np.log(0.1), np.log(10.0)],
                           'mean': [-5, 5],
                           'rel_amp': [0.1, 10],
                           'rel_mean': [-5, 5]
                           })
else:
    test_model = dummy_statmodel()
    test_model.set_priors({"test_param": [0.5, 0.5]})

test_model.set_priors({'lag': [0, 800]})

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------

Nlags = (mymock.maxtime / np.array(mymock.cadence) ).astype(int).max() * 2
print("Doing Hessian Fitting with grid of %i lags" % Nlags)
fitting_method = hessian_scan(stat_model=test_model, Nlags=Nlags,
                              constrained_domain=False,
                              step_size=2e-4,
                              max_opt_eval=int(128),
                              opt_tol=0.1,
                              )

fitting_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(fitting_method.get_evidence())

print("Doing Priorscan Fitting")
prior_scan = prior_sampling(stat_model=test_model, Nsamples=1024)
prior_scan.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(prior_scan.get_evidence())

f, (a1, a2) = plt.subplots(2, 1, sharex=True)

a1.scatter(prior_scan.results['samples']['lag'], np.exp(prior_scan.results['log_density']), label='sampling')
a1.scatter(fitting_method.results['scan_peaks']['lag'], np.exp(fitting_method.results['log_evidences']),
           label='hessian')
a1.plot(fitting_method.results['scan_peaks']['lag'], np.exp(fitting_method.results['log_evidences']),
           label='hessian')

a2.scatter(prior_scan.results['samples']['lag'], prior_scan.results['log_density'], label='sampling', s=4)
a2.scatter(fitting_method.results['scan_peaks']['lag'], fitting_method.results['log_evidences'], label='hessian', s = 4)
a2.legend()
a2.set_ylim(fitting_method.results['log_evidences'].min() + 100, fitting_method.results['log_evidences'].max()+10)

f.supylabel("Marginal Likelihood")
f.supxlabel("Lag (Days)")

for a in (a1, a2):
    a.axvline(mymock.lag, c='k', ls='--')
    a.grid()

plt.show()
