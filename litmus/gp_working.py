"""
gp_working.py

Contains all interfacing with the tinyGP gaussian process modelling package

Multi-band kernel adapated from:
    "Gaussian Process regression for astronomical time-series"
    Aigrain & Foreman-Mackey, 2022:
    https://arxiv.org/abs/2209.08940

HM 2024
"""

# ============================================
# IMPORTS
from copy import deepcopy as copy

import numpy as np

import jax
import jax.numpy as jnp

import tinygp
from tinygp import GaussianProcess
from typing import Any
from nptyping import NDArray

from litmus._utils import *
from litmus.lightcurve import lightcurve


# ============================================
# Likelihood function working
def mean_func(means, Y) -> NDArray([Any], float):
    """
    DEPRECATED - means are subtracted in the model now
    Utitlity function to take array of constants and return as gp-friendly functions

    """
    t, band = Y
    return (means[band])


@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    """
    Multi-band GP kernel that knows how to scale GP to output amplitudes
    """
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, Y) -> float:
        """
        Extracts the time value from the (time,band) coordinate so the GP can interpret the ordering of points
        in multiple bands
        """
        t, band = Y
        return t

    def observation_model(self, Y) -> float:
        """
        Scales the prediction for each band by their respective band amplitude in the predicted model
        """
        t, band = Y
        return self.amplitudes[band] * self.kernel.observation_model(t)


def build_gp(T: [float], Y: [float], diag: [[float]], bands: [int], tau: float, amps: [float], means: [float],
             basekernel=tinygp.kernels.quasisep.Exp) -> GaussianProcess:
    """
    Constructs the tinyGP gaussian process for use in numpyro sampling
    TODO: update this documentation. Possibly change to dict input

    :param data:        Banded lc as dictionary of form {T,Y,E,bands}
    :param params:      Parameters to build the gp from as dictionary
    :param basekernel:  Base gaussian kernel to use. Defaults to exponential
    :return:            Returns tinygp gp object and jnp.array of data sorted by lag-corrected time
    """

    # Create GP kernel with Multiband
    multi_kernel = Multiband(
        kernel=basekernel(scale=tau),
        amplitudes=amps,
    )

    # Mean functions for offsetting signals
    meanf = lambda X: mean_func(means, X)

    # Construct GP object and return
    gp = GaussianProcess(
        multi_kernel,
        (T, bands),
        diag=diag,
        mean=meanf
    )
    return (gp)


#-------------------
if __name__=="__main__":
    print(":)")
