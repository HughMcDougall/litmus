'''
Contains fitting procedures to be executed by the litmus class object

HM 24
'''

# ============================================
# IMPORTS
import numpyro
from numpyro import distributions as dist
from tinygp import GaussianProcess
from jax.random import PRNGKey

import jax
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro import infer

import jax.numpy as jnp
import numpy as np

from models import _default_config

from functools import partial

_default_config |={
    'Nboot': 1_000,
    'NLagGrid': 512
}

# ============================================
#

def ICCF(T,Y,E,bands, config_params = _default_config):
    Nlags = config_params['NLagGrid']
    Nboot = config_params['Nboot']

    # Un-sort Data
    I1 = jnp.where(bands == 0)[0]
    I2 = jnp.where(bands == 1)[0]
    T1, T2 = T[I1], T[I2]
    Y1, Y2 = Y[I1], Y[I2]
    E1, E2 = E[I1], E[I2]
    N1, N2 = len(I1), len(I2)

    lags = jnp.linspace(config_params['lag'][0], config_params['lag'][1], Nlags)

    E1_grid = np.tile(E1, (Nboot,1)) * np.random.randn(Nboot,N1)
    E2_grid = np.tile(E2, (Nboot,1)) * np.random.randn(Nboot,N2)

    PHI, PHI_E = _ICCF_JAX(T1,T2,Y1,Y2,E1_grid,E2_grid,Nboot,lags)

    return(lags, PHI, PHI_E)
@partial(jax.jit, static_argnames=['Nboot'])
def _ICCF_JAX(lag, T1, T2, Y1, Y2, E1, E2, Nboot, T_interp):
    PHI_BOOT = jnp.zeros(Nboot)
    for j in range(Nboot):
        Y1_interp = jnp.interp(T_interp, T1, fp=Y1, left=0, right=0)
        Y2_interp = jnp.interp(T_interp-lag, T1, fp=Y1, left=0, right=0)
        PHI_BOOT[j] = jnp.corrcoef(x=Y1_interp, y = Y2_interp)

    return (PHI_BOOT)



# ============================================
# Testing
if __name__== "__main__":
    import matplotlib.pyplot as plt

    print(":)")
    T1 = np.linspace(0,4*np.pi,128)
    T2 = np.linspace(0, 4 * np.pi, 128)
    Y1 = np.sin(T1) * np.sqrt(2)
    E1 = (np.random.poisson(100,size=len(T1)) / 100) * 1
    E2 = (np.random.poisson(100,size=len(T2)) / 100) * 1
    Y1+=np.random.randn(len(E1))*E1
    Y2 = np.sin(T2 - 0.25) + 5 + np.random.randn(len(E2))*E2

    bands = np.concatenate([np.zeros_like(T1),np.ones_like(T2)]).astype(int)
    T = np.concatenate([T1, T2])
    Y = np.concatenate([Y1, Y2])
    E = np.concatenate([E1, E2])

    #lags, PHI, PHI_E = ICCF(T, Y, E, bands, config_params=_default_config)

    _ICCF_JAX(0,T1,T2,Y1,Y2,E1,E2,100,jnp.linspace(-10,10,512))
