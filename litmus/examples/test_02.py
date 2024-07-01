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

mymock=mock_C()
mock_array = []
for i, Escale in enumerate([0.01, 0.05, 0.1, 0.25]):
    mymock.E = [Escale,Escale]
    mymock = mymock(seed=i)
    plt.sca(a[i])
    mymock.plot(axis=plt.gca())

    a[i].grid()

    mock_array.append(mymock)

a[0].legend()
f.suptitle("True_lag = %.0f days" % mymock.lag)
f.supxlabel("Time, Days")
plt.xlim(0, mymock.lc.T.max())
f.tight_layout()
plt.show()

# --------------------------------
test_model = GP_simple()
test_model.set_priors({'logtau': [np.log(mymock.tau), np.log(mymock.tau)],
                       'logamp': [0, 0],
                       'mean': [0, 0],
                       'rel_amp': [1, 1],
                       'rel_mean': [0, 0]
                       })

test_model.set_priors({'lag': [0, 1000]})

prior_samp_method = prior_sampling(stat_model=test_model, Nsamples=1024)
plt.figure()

f, (a1, a2) = plt.subplots(2, 1)

print("Doing prior samping")
for i, mymock in enumerate(mock_array):
    print(i, end='\t')
    prior_samp_method.fit(lc_1=mymock.lc_1, lc_2=mymock.lc_2, seed=0)
    sample_lags = prior_samp_method.results['samples']['lag']
    sample_loglikes = prior_samp_method.results['log_density']
    Z = prior_samp_method.get_evidence()[0]

    I = np.argsort(sample_lags)

    a1.plot(sample_lags[I], sample_loglikes[I] - np.log(Z))
    a2.plot(sample_lags[I], np.exp(sample_loglikes[I] - np.log(Z)))

print("Done.")
for a in (a1, a2):
    a.axvline(mymock.lag, ls='--', c='k', zorder=-1, alpha=0.5)
    a.grid()

a1.set_ylabel("Likelihood")
a2.set_ylabel("Log-Likelihood")

a2.legend()
plt.tight_layout()
plt.show()
