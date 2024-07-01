'''
A quick and nasty test file to confirm everything's working

HM 24
'''

import os, sys

import matplotlib.pyplot as plt

sys.path.append("../../")
import litmus

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

from models import stats_model, dummy_statmodel, GP_simple
from fitting_methods import fitting_procedure, nested_sampling, prior_sampling, hessian_scan
from chainconsumer import ChainConsumer

from mocks import mock, mock_A, mock_B, mock_C

# ============================================

f, a = plt.subplots(4, 1, figsize=(5, 8), sharex=True, sharey=True)

mymock=mock_B(seed=0)

# --------------------------------
if True:
    test_model = GP_simple()
    test_model.set_priors({'logtau': [np.log(mymock.tau*0.1), np.log(mymock.tau*10)],
                           'logamp': [0, 0],
                           'mean': [0, 0],
                           'rel_amp': [1, 1],
                           'rel_mean': [0, 0]
                           })
else:
    test_model = dummy_statmodel()
    test_model.set_priors({"test_param":[0.5,0.5]})

test_model.set_priors({'lag': [0, 1000]})

data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)
starts = test_model.prior_sample() | {'lag': mymock.lag}

finals = test_model.scan(start_params=starts, optim_params=['logtau'], data=data)
for key in starts.keys():
    print(key, starts[key], finals[key])
print(starts, finals)

print("Doing Fitting")
#fitting_method = hessian_scan(stat_model=test_model, Nlags=128)
#fitting_method.fit(lc_1 = mymock.lc_1, lc_2=mymock.lc_2)

