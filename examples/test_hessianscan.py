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

mymock = mock(cadence=[100, 300], E=[0.1, 0.1], season=None)
mymock.plot(true_args={'alpha': 0.0})
plt.show()

# --------------------------------

# Switch for GP simple or dummy model
if True:
    test_model = GP_simple()
    '''
    test_model.set_priors({'logtau': [np.log(mymock.tau * 0.1), np.log(mymock.tau * 10)],
                           'logamp': [0, 0],
                           'mean': [0, 0],
                           'rel_amp': [1, 1],
                           'rel_mean': [0, 0]
                           })
    '''
else:
    test_model = dummy_statmodel()
    test_model.set_priors({"test_param": [0.5, 0.5]})

test_model.set_priors({'lag': [0, 1000]})

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)

# ---------------------------------------------------------------------
# Check to make sure scan is working reasonably well
'''
starts = test_model.prior_sample() | {'lag': mymock.lag}
finals = test_model.scan(start_params=starts, optim_params=['logtau'], data=data)
print("After running a scan at the true lag, parameters are:")
for key in starts.keys():
    print(key, starts[key], finals[key])
print(starts, finals)
'''

print("Doing Hessian Fitting")
fitting_method = hessian_scan(stat_model=test_model, Nlags=16, constrained_domain = False)
fitting_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(fitting_method.get_evidence())

print("Doing Priorscan Fitting")
prior_scan = prior_sampling(stat_model=test_model, Nsamples=1024)
prior_scan.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2)
print("Evidences are:")
print(prior_scan.get_evidence())

plt.figure()
plt.scatter(prior_scan.results['samples']['lag'], prior_scan.results['log_density'], label='sampling')
plt.scatter(fitting_method.results['scan_peaks']['lag'], fitting_method.results['log_evidences'], label='hessian')
plt.legend()
#plt.ylim(-20,-7)

print("Log gap approx, %.2f" %(fitting_method.results['log_evidences'].max() - prior_scan.results['log_density'].max()))
plt.show()
