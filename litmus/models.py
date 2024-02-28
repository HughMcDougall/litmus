'''
Contains NumPyro generative models.

HM 24
'''

# ============================================
# IMPORTS
import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from jax.random import PRNGKey
import jax.numpy as jnp
import tinygp
from gp_working import *
# ============================================
#


def GP_single(T,Y,E,bands,
              basekernel = tinygp.kernels.quasisep.Exp,
              config_params = {}):
    '''
    :param T:
    :param Y:
    :param E:
    :param bands:
    :param basekernel:
    :param config_params:
    :return:
    '''

    # Sample distributions
    logtau   = numpyro.sample('logtau', dist.Uniform(config_params['logtau'][0], config_params['logtau'][1]))
    logamp   = numpyro.sample('logamp', dist.Uniform(config_params['logamp'][0], config_params['logamp'][1]))
    rel_amp  = numpyro.sample('rel_amp', dist.Uniform(0.0, config_params['rel_amp']))
    mean     = numpyro.sample('mean', dist.Uniform(config_params['mean'][0], config_params['mean'][1]))
    rel_mean = numpyro.sample('rel_mean', dist.Uniform(config_params['rel_mean'][0], config_params['rel_mean'][1]))


    amp, tau = jnp.exp(amp), jnp.exp(logtau)

    diag = jnp.diag(jnp.square(E))

    build_gp(T,Y,diag,bands,tau,amps,mean,basekernel=basekernel)