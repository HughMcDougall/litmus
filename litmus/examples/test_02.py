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

from models import stats_model
from chainconsumer import ChainConsumer
# ============================================

mock = mock_B(seed=0)

plt.figure()
mock.plot(axis = plt.gca())
plt.grid()
plt.legend()
plt.show()