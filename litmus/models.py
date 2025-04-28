"""
Contains NumPyro generative models.

HM 24
"""

# ============================================
# IMPORTS

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import sys

import numpy as np
import scipy

import jax.scipy.optimize
from jax.random import PRNGKey
import jax.numpy as jnp
import jaxopt

import numpyro
from numpyro.distributions import MixtureGeneral
from numpyro import distributions as dist
from numpyro import handlers

import tinygp
from tinygp import GaussianProcess

from litmus.lightcurve import lightcurve
import litmus.gp_working as gpw
import litmus._utils as _utils
import litmus._types as _types
from litmus.logging import logger

import contextlib

import os


def quickprior(targ, key):
    p = targ.prior_ranges[key]
    distrib = dist.Uniform(float(p[0]), float(p[1])) if p[0] != p[1] else dist.Delta(float(p[0]))
    out = numpyro.sample(key, distrib)
    return out


# ============================================
#

# Default prior ranges. Kept in one place as a convenient storage area
_default_config = {
    'logtau': [0.0, 10.0],
    'logamp': (-3.0, 3.0),
    'rel_amp': [0.0, 10.0],
    'mean': [-20.0, +20.0],
    'rel_mean': [-10.0, +10.0],
    'lag': [0.0, 1000.0],
    'outlier_spread': [0.0, 10.0],
    'outlier_frac': [0.0, 0.25],
}


# ============================================
# Base Class

class stats_model(logger):
    """
    Base class for bayesian generative models. Includes a series of utilities for evaluating likelihoods, gradients etc.,
    as well as various

    On init, takes dict `prior_ranges' of the uniform boundaries of the parameter priors, or a single (float/int)
    value if the value is fixed.
    
    Also takes logging arg from the litmus.logging.logger object.
    """

    def __init__(self, prior_ranges=None,
                 out_stream=sys.stdout,
                 err_stream=sys.stderr,
                 verbose=True,
                 debug=False,
                 warn=1
                 ):
        logger.__init__(self, out_stream=out_stream, err_stream=err_stream, verbose=verbose, debug=debug, warn=warn)

        self._protected_keys = []

        # Setting prior boundaries
        if not hasattr(self, "_default_prior_ranges"):
            self._default_prior_ranges: dict[str, list[float, float]] = {
                'lag': _default_config['lag'],
            }

        # ------------------
        # Wrapped function signature declarations for type hinting
        self._log_density_jit: _types.Callable[[dict, _types.Any], float]
        self._log_density_uncon_jit: _types.Callable[[dict, _types.Any], float]
        self._log_likelihood_jit: _types.Callable[[dict, _types.Any], float]
        self._log_prior_jit: _types.Callable[[dict], float]

        self._log_density_grad: _types.Callable[[dict, _types.Any], _types.ArrayM]
        self._log_density_uncon_grad: _types.Callable[[dict, _types.Any], _types.ArrayM]
        self._log_likelihood_grad: _types.Callable[[dict, _types.Any], _types.ArrayM]
        self._log_prior_grad: _types.Callable[[dict], _types.ArrayM]

        self._log_density_hess: _types.Callable[[dict, _types.Any], _types.ArrayMxM]
        self._log_density_uncon_hess: _types.Callable[[dict, _types.Any], _types.ArrayMxM]
        self._log_likelihood_hess: _types.Callable[[dict, _types.Any], _types.ArrayMxM]
        self._log_prior_hess: _types.Callable[[dict], _types.ArrayMxM]
        # ------------------

        # Attributes
        self.prior_ranges: dict[str, list[float, float]] = {} | self._default_prior_ranges
        """Keyed dict like {key: [max,min] } of bounds for parameter uniform priors"""
        self.prior_volume = 1.0
        """Volume of the prior, i.e. prod(max_i-min_i) for in in params"""
        self.name = type(self).__name__
        """Name of the model for print strings"""

        # Update with args
        self.set_priors(self._default_prior_ranges | prior_ranges) if prior_ranges is not None else self.set_priors(
            self._default_prior_ranges)

        self._prep_funcs()

    def __setattr__(self, key, value):
        if key == "_default_prior_ranges" and hasattr(self, "_default_prior_ranges"):
            super().__setattr__(key, value | self._default_prior_ranges)
        elif key == "_protected_keys" and hasattr(self, "_protected_keys"):
            super().__setattr__(key, value + self._protected_keys)
        else:
            super().__setattr__(key, value)

        if hasattr(self, "_protected_keys") and key in self._protected_keys:
            self._prep_funcs()

    def _prep_funcs(self):
        """
        Internal trigger to re-compile functions. Needs to run anytime model priors are changed
        """
        # --------------------------------------
        ## Create jitted, vmapped and grad/hessians of all density functions

        for func in [self._log_density, self._log_density_uncon, self._log_prior, self._log_likelihood]:
            name = func.__name__
            # unpacked_func = _utils.pack_function(func, packed_keys=self.paramnames())

            # Take grad, hessian and jit
            jitted_func = jax.jit(func)
            graded_func = jax.jit(jax.grad(func, argnums=0))
            hessed_func = jax.jit(jax.hessian(func, argnums=0))

            jitted_func.__doc__ = func.__doc__ + ", jitted version"
            graded_func.__doc__ = func.__doc__ + ", grad'd and jitted version"
            hessed_func.__doc__ = func.__doc__ + ", hessian'd and jitted version"

            # todo - add vmapped versions to these as well, and possibly leave raw un-jitted calls for better performance down the track
            # todo - Maybe have packed fuction calls for easier math-ing on the hessian? Probably not. We can unpack the interesting ones later

            # Set attributes
            self.__setattr__(name + "_jit", jitted_func)
            self.__setattr__(name + "_grad", graded_func)
            self.__setattr__(name + "_hess", hessed_func)
        # ------------------
        # Aux functions
        self.gen_lightcurve = self._gen_lightcurve

        ## --------------------------------------

    def set_priors(self, prior_ranges: dict) -> None:
        """
        Sets the stats model prior ranges for uniform priors. Does some sanity checking to avoid negative priors.

        :param prior_ranges: keyed dict of {key: [minval, maxval]} for ranges or {key: fixedval} for fixed values
        """

        badkeys = [key for key in prior_ranges.keys() if key not in self._default_prior_ranges.keys()]

        for key, val in zip(prior_ranges.keys(), prior_ranges.values()):
            if key in badkeys:
                continue

            if _utils.isiter(val):
                a, b = val
            else:
                try:
                    a, b = val, val
                except:
                    raise "Bad input shape in set_priors for key %s" % key  # todo - make this go to std.err

            self.prior_ranges[key] = [float(a), float(b)]

        # Calc and set prior volume
        # Todo - Make this more general. Revisit if we separate likelihood + prior
        prior_volume = 1.0
        for key in self.prior_ranges:
            a, b = self.prior_ranges[key]
            if b != a:
                prior_volume *= b - a
        self.prior_volume = prior_volume

        self._prep_funcs()

        return

    # --------------------------------
    # MODEL FUNCTIONS
    def prior(self) -> [float, ]:
        """
        A NumPyro callable prior.
        :returns: Values of the parameters as sampled from the prior
        """
        lag = numpyro.sample('lag', dist.Uniform(self.prior_ranges['lag'][0], self.prior_ranges['lag'][1]))
        return lag

    def model_function(self, data):
        """
        A NumPyro callable function. Does not return
        :param data: Data to condition the model on
        """
        lag = self.prior()

    def lc_to_data(self, lc_1: lightcurve, lc_2: lightcurve) -> dict:
        """
        Converts light-curves into the data format required for the model. For most models this will return as some sort
        of sorted dictionary.
        :param lc_1: First lightcurve object
        :param lc_2: Second lightcurve object
        :return: Varies from model to model, by default will be a keyed dict:
            {'T': Time values of observations series,
             'Y': Signal strength values of observations series,
             'E': Uncertainty values of values in Y,
             'bands': int array identifying which lightcurve (0,1) that the observations belong to
            }
        """

        T = jnp.array([*lc_1.T, *lc_2.T])
        Y = jnp.array([*lc_1.Y, *lc_2.Y])
        E = jnp.array([*lc_1.E, *lc_2.E])
        bands = jnp.array([*np.zeros(lc_1.N), *np.ones(lc_2.N)]).astype(int)

        I = T.argsort()

        T, Y, E, bands = T[I], Y[I], E[I], bands[I]

        data = {'T': T,
                'Y': Y,
                'E': E,
                'bands': bands
                }

        return data

    # --------------------------------
    # Parameter transforms and other utils
    def to_uncon(self, params) -> dict[str, float]:
        """
        Converts model parametes from "real" constrained domain values into HMC friendly unconstrained values.

        :param params: keyed dict of parameters in constrained domain
        :return: keyed dict of parameters in unconstrained domain
        """
        out = numpyro.infer.util.unconstrain_fn(self.prior, params=params, model_args=(), model_kwargs={})
        return out

    def to_con(self, params) -> dict[str, float]:
        """
        Converts model parametes back into "real" constrained domain values.
        :param params: keyed dict of parameters in unconstrained domain
        :return: keyed dict of parameters in constrained domain
        """
        out = numpyro.infer.util.constrain_fn(self.prior, params=params, model_args=(), model_kwargs={})
        return out

    def uncon_grad(self, params) -> float:
        """
        Evaluates the log of det(Jac) by evaluating pi(x) and pi'(x').
        Used for correcting integral elements between constrained and unconstrained space

        :param params: Model parameters in constrained domain
        :return: float of det(Jacobian)
        """
        con_dens = numpyro.infer.util.log_density(self.prior, (), {}, params)[0]

        up = self.to_uncon(params)
        uncon_dens = -numpyro.infer.util.potential_energy(self.prior, (), {}, up)
        out = con_dens - uncon_dens
        return out

    def uncon_grad_lag(self, params) -> float:
        """
        Returns the log-jacobian correction for the constrained / unconstrained correction for the lag parameter
        Assumes a uniform distribution for the lag prior

        :param params: Model parameters in constrained domain
        :return: float of det(Jacobian) for lag_uncon <-> lag_con
        """

        from numpyro.infer.util import transform_fn

        if 'lag' not in self.paramnames(): return 0
        if np.ptp(self.prior_ranges['lag']) == 0: return 0

        lagdist = dist.Uniform(*self.prior_ranges['lag'])
        lag_con = params['lag']

        transforms = {"lag": numpyro.distributions.biject_to(lagdist.support)}

        def tform(x):
            out = transform_fn(transforms, params | {'lag': x}, invert=True)['lag']
            return out

        tform = jax.grad(tform)
        out = np.log(abs(tform(lag_con)))
        return out

    def paramnames(self) -> [str]:
        """
        Returns the names of all model parameters. Purely for brevity of code.

        :return: list of param names in order listed in prior_ranges
        """
        return list(self.prior_ranges.keys())

    def fixed_params(self) -> [str]:
        """
        Returns the names of all fixed model parameters. Purely for brevity.

        :return: list of param names in order listed in prior_ranges
        """
        is_fixed = {key: np.ptp(self.prior_ranges[key]) == 0 for key in self.prior_ranges.keys()}
        out = [key for key in is_fixed.keys() if is_fixed[key]]
        return out

    def free_params(self) -> [str]:
        """
        Returns the names of all free model parameters. Purely for brevity of code.

        :return: list of param names in order listed in prior_ranges
        """
        is_fixed = {key: np.ptp(self.prior_ranges[key]) == 0 for key in self.prior_ranges.keys()}
        out = [key for key in is_fixed.keys() if not is_fixed[key]]
        return out

    def dim(self) -> int:
        """
        Quick and easy call for the number of model parameters.

        :return: number of model parameters as int
        """
        return len(self.free_params())

    # --------------------------------
    # Un-Jitted / un-vmapped likelihood calls
    """
    Functions in this sector are in their basic form. Those with names appended by '_forgrad' accept inputs as arrays
    """

    def _log_density(self, params: dict, data):
        """
        Constrained space un-normalized posterior log density

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :return: log density as float
        """
        out = \
            numpyro.infer.util.log_density(self.model_function, params=params, model_args=(),
                                           model_kwargs={'data': data})[
                0]
        return out

    def _log_likelihood(self, params, data) -> float:
        """
        WARNING! This function won't work if your model has more than one observation site!
        Constrained space un-normalized posterior log likelihood

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :return: log likelihood as float
        """

        out = self._log_density(params, data) - self._log_prior(params, data)
        return out

    def _log_density_uncon(self, params, data) -> float:
        """
        Unconstrained space un-normalized posterior log density
        :param params: model params in unconstrained space as keyed dict
        :param data: data to condition the model on
        :return: log likelihood as float
        """
        out = -numpyro.infer.util.potential_energy(self.model_function, params=params, model_args=(),
                                                   model_kwargs={'data': data})
        return out

    def _log_prior(self, params, data=None) -> float:
        """
        Model prior density in unconstrained space
        """
        out = numpyro.infer.util.log_density(self.prior, (), {}, params)[0]
        return out

    # --------------------------------
    # Wrapped Function Evaluations
    def log_density(self, params, data, use_vmap=False) -> _types.ArrayN:
        """
        Returns the log density of the joint distribution at some constrained space position 'params' and conditioned
        on some 'data'. data must match the output of the model's lc_to_data(), and params is either a keyed dict of
        parameter values or a key dict of arrays of values.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)

        :return: Returns as array of floats

        """

        if _utils.isiter_dict(params):
            N = _utils.dict_dim(params)[1]
            out = np.zeros(N)
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                out[i] = self._log_density_jit(p, data)
        else:
            out = np.array([self._log_density_jit(params, data)])

        return out

    def log_likelihood(self, params, data, use_vmap=False) -> _types.ArrayN:
        """
        Returns the log likelihood at some constrained space position 'params' and conditioned
        on some 'data'. data must match the output of the model's lc_to_data(), and params is either a keyed dict of
        parameter values or a key dict of arrays of values.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)

        :return: Returns as array of floats
        """
        if _utils.isiter_dict(params):
            N = _utils.dict_dim(params)[1]
            out = np.zeros(N)
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                out[i] = self._log_likelihood(p, data)
        else:
            out = self._log_likelihood(params, data)

        return out

    def log_density_uncon(self, params, data, use_vmap=False) -> _types.ArrayN:
        """
        Returns the log density of the joint distribution at some unconstrained space position 'params' and conditioned
        on some 'data'. data must match the output of the model's lc_to_data(), and params is either a keyed dict of
        parameter values or a key dict of arrays of values.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)

        :return: Returns as array of floats
        """

        if _utils.isiter_dict(params):
            N = _utils.dict_dim(params)[1]
            out = np.zeros(N)
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                out[i] = self._log_density_uncon_jit(p, data)
        else:
            out = self._log_density_uncon_jit(params, data)

        return out

    def log_prior(self, params, data=None, use_vmap=False) -> _types.ArrayN:
        """
        Returns the log density of the prior  at some constrained space position 'params'
        Params is either a keyed dict of parameter values or a key dict of arrays of values.
        Returns as array of floats
        use_vmap currently not implemented with no side effect
        """

        if _utils.isiter_dict(params):
            N = _utils.dict_dim(params)[1]
            out = np.zeros(N)
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                out[i] = self._log_prior_jit(p)
        else:
            out = self._log_prior_jit(params)

        return out

    # --------------------------------
    # Wrapped Grad evaluations
    def log_density_grad(self, params, data, use_vmap=False, keys=None) -> dict[str, float]:
        """
        Returns the gradient of the log density of the joint distribution at some constrained space position 'params',
        conditionded on some 'data' matching the format of the model's lc_to_data() output.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: list of keys / parameters to take the gradient over

        :return: Returns as keyed dict of grads along each axsi or keyed dict of array of similar values

        """

        if _utils.isiter_dict(params):
            m, N = _utils.dict_dim(params)
            out = {key: np.zeros([N]) for key in params.keys()}
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                grads = self._log_density_grad(p, data)
                for key in params.keys():
                    out[key][i] = grads[key]
        else:
            out = self._log_density_grad(params, data)

        return out

    def log_density_uncon_grad(self, params, data, use_vmap=False, keys=None, asdict=False) -> float:
        """
        Returns the gradient of the log density of the joint distribution at some unconstrained space position 'params',
        conditionded on some 'data' matching the format of the model's lc_to_data() output.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: list of keys / parameters to take the gradient over

        :return: Returns as keyed dict of grads along each axsi or keyed dict of array of similar values
        """

        if _utils.isiter_dict(params):
            m, N = _utils.dict_dim(params)
            out = {key: np.zeros([N]) for key in params.keys()}
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                grads = self._log_density_uncon_grad(p, data)
                for key in params.keys():
                    out[key][i] = grads[key]
        else:
            out = self._log_density_uncon_grad(params, data)

        return out

    def log_prior_grad(self, params, data=None, use_vmap=False, keys=None) -> dict[str, float]:
        """
        Returns the gradient of the log prior of the prior at some constrained space position 'params'
        Params is either a keyed dict of parameter values or a key dict of arrays of values.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: list of keys / parameters to take the gradient over

        :return: Returns as keyed dict of grads along each axsi or keyed dict of array of similar values
        """

        if _utils.isiter(params):
            m, N = _utils.dict_dim(params)
            out = np.zeros(N)
            for i in range(N):
                p = {key: params[key][i] for key in params.keys()}
                out[i, :] = self._log_prior_grad(p)
        else:
            out = self._log_prior_grad(params)

        return out

    # --------------------------------
    # Wrapped Hessian evaluations
    def log_density_hess(self, params, data, use_vmap=False, keys=None) -> _types.ArrayNxMxM:
        """
        Returns the hessian matrix of the log joint distribution at some constrained space position 'params',
        conditioned on some 'data' matching the output of the model's lc_to_data() output.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: The params to slice and sort the hessian matrices.

        :return: Returns in order / dimension: [num param sites, num keys, num keys]
        """

        if keys is None: keys = params.keys()

        if _utils.isiter_dict(params):
            m, N = _utils.dict_dim(params)
            m = len(keys)
            out = np.zeros([N, m, m])
            for i in range(N):
                p = {key: params[key][i] for key in keys}
                hess_eval = self._log_density_hess(p, data)
                for j, key1 in enumerate(keys):
                    for k, key2 in enumerate(keys):
                        out[i, j, k] = hess_eval[key1][key2]
        else:
            m = len(keys)
            out = np.zeros([m, m])
            hess_eval = self._log_density_hess(params, data)
            for j, key1 in enumerate(keys):
                for k, key2 in enumerate(keys):
                    out[j, k] = hess_eval[key1][key2]

        return out

    def log_density_uncon_hess(self, params, data, use_vmap=False, keys=None) -> _types.ArrayNxMxM:
        """
        Returns the hessian matrix of the log joint distribution at some unconstrained space position 'params',
        conditioned on some 'data' matching the output of the model's lc_to_data() output.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: The params to slice and sort the hessian matrices.

        :return: Returns in order / dimension: [num param sites, num keys, num keys]
        """

        if keys is None: keys = params.keys()

        if _utils.isiter_dict(params):
            m, N = _utils.dict_dim(params)
            m = len(keys)
            out = np.zeros([N, m, m])
            for i in range(N):
                p = {key: params[key][i] for key in keys}
                hess_eval = self._log_density_uncon_hess(p, data)
                for j, key1 in enumerate(keys):
                    for k, key2 in enumerate(keys):
                        out[i, j, k] = hess_eval[key1][key2]
        else:
            m = len(keys)
            out = np.zeros([m, m])
            hess_eval = self._log_density_uncon_hess(params, data)
            for j, key1 in enumerate(keys):
                for k, key2 in enumerate(keys):
                    out[j, k] = hess_eval[key1][key2]

        return out

    def log_prior_hess(self, params, data=None, use_vmap=False, keys=None) -> _types.ArrayNxMxM:
        """
        Returns the hessian matrix of the log prior of the prior at some constrained space position 'params'
        Params is either a keyed dict of parameter values or a key dict of arrays of values.

        :param params: model params in constrained space as keyed dict
        :param data: data to condition the model on
        :param use_vmap: whether to use vmap instead of unconstrained space (not implemented)
        :param keys: The params to slice and sort the hessian matrices.

        :return: Returns in order / dimension: [num param sites, num keys, num keys]
        """

        if keys is None: keys = params.keys()

        if _utils.isiter_dict(params):
            m, N = _utils.dict_dim(params)
            m = len(keys)
            out = np.zeros([N, m, m])
            for i in range(N):
                p = {key: params[key][i] for key in keys}
                hess_eval = self._log_prior_hess(p)
                for j, key1 in enumerate(keys):
                    for k, key2 in enumerate(keys):
                        out[i, j, k] = hess_eval[key1][key2]
        else:
            m = len(keys)
            out = np.zeros([m, m])
            hess_eval = self._log_prior_hess(params)
            for j, key1 in enumerate(keys):
                for k, key2 in enumerate(keys):
                    out[j, k] = hess_eval[key1][key2]

        return out

    # --------------------------------
    # Wrapped evaluation utilities

    def _scanner(self, data, optim_params=None, use_vmap=False, optim_kwargs={}, return_aux=False, precondition='diag'):
        """
        Creates a black-box jitted optimizer for when we want to perform many scans in a row
        """

        # Convert to unconstrainedc domain
        if optim_params is None:
            optim_params = [name for name in self.paramnames() if
                            self.prior_ranges[name][0] != self.prior_ranges[name][1]]

        # ---------------------------
        def converter(start_params):
            start_params_uncon = self.to_uncon(start_params)

            # Get all split into fixed and free params
            x0 = jnp.array([start_params_uncon[key] for key in optim_params])
            y0 = {key: start_params_uncon[key] for key in start_params_uncon.keys() if key not in optim_params}

            return x0, y0

        def deconverter(x, y):
            x = {key: x[i] for i, key in enumerate(optim_params)}
            x = x | y  # Adjoin the fixed values
            opt_params = self.to_con(x)

            return opt_params

        def runsolver(solver, start_params, aux: bool = False):

            # Main Loop
            # todo - change this to a .update loop to allow us to pass the state
            x0, y0 = converter(start_params)
            with _utils.suppress_stdout():  # TODO - Supressing of warnings, should be patched in newest jaxopt
                xopt, state = solver.run(x0, y0, data)
            out_params = deconverter(xopt, y0)

            # ------------
            # Returns
            if aux == False:
                return out_params
            else:
                aux_data = {
                    'H': state.H,
                    'err': state.error,
                    'grad': state.grad,
                    'val': state.value,
                    'stepsize': state.stepsize,
                    'iter_num': state.iter_num,
                }
                return out_params, aux_data

        def runsolver_jit(solver, start_params, state):
            x0, y0 = converter(start_params)
            outstate = _utils.copy(state)

            i = 0

            while i == 0 or (outstate.error > solver.tol and i < solver.maxiter):
                with _utils.suppress_stdout():  # TODO - Supressing of warnings, should be patched in newest jaxopt
                    x0, outstate = solver.update(x0, outstate, y0, data)
                i += 1
            out_params = deconverter(x0, y0)

            aux_data = {
                'H': state.H,
                'err': state.error,
                'grad': state.grad,
                'val': state.value,
                'stepsize': state.stepsize,
            }
            return out_params, aux_data, outstate

        # ---------------------------

        optimizer_args = {
            'stepsize': 0.0,
            'min_stepsize': 1E-5,
            'increase_factor': 1.2,
            'maxiter': 1024,
            'linesearch': 'backtracking',
            'verbose': False,
        }

        optimizer_args |= optim_kwargs

        x0, y0 = converter(self.prior_sample())
        # Make a jaxopt friendly packed function
        optfunc = _utils.pack_function(self._log_density_uncon,
                                       packed_keys=optim_params,
                                       fixed_values={},
                                       invert=True)

        # Build and run an optimizer
        solver = jaxopt.BFGS(fun=optfunc,
                             value_and_grad=False,
                             jit=True,
                             **optimizer_args
                             )

        if self.debug:
            try:
                self.msg_debug("Creating and testing solver...")
                init_state = solver.init_state(x0, data=data)
                with _utils.suppress_stdout():  # TODO - Supressing of warnings, should be patched in newest jaxopt
                    solver.update(params=x0, state=init_state, y0=y0, data=data)
                self.msg_debug("Jaxopt solver created and running fine")
            except:
                self.msg_err("Something wrong in creation of jaxopt solver!", lvl=0)

        # Return

        if return_aux:
            return solver, runsolver, [converter, deconverter, optfunc, runsolver_jit]
        else:
            return solver, runsolver

    def scan(self, start_params: dict[str:float], data: _types.Any, optim_params: [str] = None, use_vmap: bool = False,
             optim_kwargs: dict[str:float] = {},
             precondition: _types.Literal["cholesky", "eig", "half-eig", "diag", "none"] = 'diag',
             solver: _types.Literal["BFGS", "GradientDescent"] = "BFGS") -> dict[str, float]:
        """
        Beginning at position 'start_params', optimize parameters in 'optim_params' to find maximum.
        optim_kwargs will overwrite defaults and be passed directly to jaxopt.BFGS object

        Currently using jaxopt with optim_kwargs:
            'stepsize': 0.0,
            'min_stepsize': 1E-5,
            'increase_factor': 1.2,
            'maxiter': 256,
            'linesearch': 'backtracking',
            'verbose': False,

        :param start_params: Position in constrained parameter spac to begin the search at
        :param data: data to condition the model on
        :param optim_params: parameters to optimize over
        :param use_vmap: Whether to use jax vmapping over multiple start_params (not implemented)
        :param optim_kwargs: kwargs for the jaxopt optimiser
        :param precondition: Type of preconditioning to use in optimisation. Should be "cholesky", "eig", "half-eig", "diag" or "none"
        :param solver: Type of jaxopt solver to use. Defaults to "BFGS". Allowed solvers are "BFGS" or "GradientDescent".

        :returns: Keyed dict of optimised params
        """

        assert solver in ["BFGS", "GradientDescent"]

        self.msg_debug("Doing scan with solver:\t %s" % solver)

        if solver == "BFGS":
            optimizer_args = {
                'stepsize': 0.0,
                'min_stepsize': 1E-5,
                'increase_factor': 1.2,
                'maxiter': 256,
                'linesearch': 'backtracking',
                'verbose': False,
            }
        elif solver == "GradientDescent":
            optimizer_args = {
                'stepsize': 0.0,
                'maxiter': 1024,
                'verbose': False,
            }
            precondition = "none"

        optimizer_args |= optim_kwargs

        # Convert to unconstrained domain
        start_params_uncon = self.to_uncon(start_params)

        self.msg_debug("Uncon params are:",
                       *["%s:\t%.2f" % (key, val) for key, val in
                         zip(start_params_uncon.keys(), start_params_uncon.values())],
                       delim="\n")

        if optim_params is None: optim_params = self.free_params()
        if len(optim_params) == 0: return start_params

        # Get all split into fixed and free params
        x0, y0 = _utils.dict_split(start_params_uncon, optim_params)
        x0 = _utils.dict_pack(x0)

        # -------------------------------------
        # Build preconditioning matrix
        H = self.log_density_uncon_hess(start_params_uncon, data, keys=optim_params)
        H *= -1

        if precondition == "cholesky":
            H = np.linalg.cholesky(np.linalg.inv(H))
            Hinv = np.linalg.inv(H)

        elif precondition == "eig":
            D, P = np.linalg.eig(H)
            if D.min() < 0:
                D[np.where(D < 0)[0].min()] = 1.0
            D, P = D.astype(float), P.astype(float)
            D **= -0.5

            H = np.dot(P, np.dot(np.diag(D), P.T))
            Hinv = np.dot(P, np.dot(np.diag(D ** -1), P.T))

        elif precondition == "half-eig":
            D, P = np.linalg.eig(H)
            if D.min() < 0:
                D[np.where(D < 0)[0].min()] = 1.0
            D, P = D.astype(float), P.astype(float)
            D **= -0.5

            H = np.dot(P, np.diag(D))
            Hinv = np.dot(np.diag(D ** -1), P.T)

        elif precondition == "diag":
            D = np.diag(H) ** -0.5
            D = np.where(D > 0, D, 1.0)
            H = np.diag(D)
            Hinv = np.diag(1 / D)

        else:
            H = np.eye(len(optim_params))
            Hinv = np.eye(len(optim_params))

        if solver == "BFGS":
            if np.any(np.isnan(H)) or np.any(np.isnan(Hinv)):
                self.msg_err("scan failed with BFGS due to broken %s scaling matrix" % precondition, lvl=1)
                return start_params
            self.msg_debug("Scaling matrix:", H)
            self.msg_debug("Inverse Scaling matrix:", Hinv)

        def optfunc(X):
            Y = jnp.dot(H, X) + x0
            params = y0 | {key: Y[i] for i, key in enumerate(optim_params)}
            out = - self._log_density_uncon(params, data)
            return out

        X0 = np.zeros_like(x0)

        self.msg_debug("At initial uncon position", X0, "with keys", optim_params, "eval for optfunc is",
                       optfunc(X0))

        assert not np.isinf(optfunc(X0)), "Something wrong with start positions in scan!"

        # =====================
        # Jaxopt Work

        # Build the optimizer
        if solver == "BFGS":
            solver = jaxopt.BFGS(fun=optfunc,
                                 value_and_grad=False,
                                 jit=True,
                                 **optimizer_args
                                 )
        elif solver == "GradientDescent":
            solver = jaxopt.GradientDescent(fun=optfunc,
                                            value_and_grad=False,
                                            jit=True,
                                            **optimizer_args
                                            )

        # Debug safety check to see if something's breaking

        if self.debug:
            self.msg_debug("Creating and testing solver...")
            try:
                init_state = solver.init_state(X0)
                with _utils.suppress_stdout():  # TODO - Supressing of warnings, should be patched in newest jaxopt
                    solver.update(params=X0, state=init_state)
                self.msg_debug("Jaxopt solver created and running fine")
            except:
                self.msg_debug("Something went wrong in when making the jaxopt optimizer. Double check your inputs.")

        with _utils.suppress_stdout():  # TODO - Supressing of warnings, should be patched in newest jaxopt
            sol, state = solver.run(init_params=X0)

        out = np.dot(H, sol) + x0

        # =====================
        # Cleanup and return
        self.msg_debug("At final uncon position", out, "with keys", optim_params, "eval for optfunc is",
                       optfunc(sol), "a change of", optfunc(sol) - optfunc(np.zeros_like(sol))
                       )
        if optfunc(sol) > optfunc(np.zeros_like(sol)):
            self.msg_err("Optimization may have diverged. Try changing the start params or use a simpler solver type",
                         lvl=1)

        # Unpack the results to a dict
        out = {key: out[i] for i, key in enumerate(optim_params)}
        out = out | y0  # Adjoin the fixed values

        # Convert back to constrained domain
        out = self.to_con(out)

        out = {key: float(val) for key, val in zip(out.keys(), out.values())}

        return out

    def laplace_log_evidence(self, params, data, integrate_axes=None, use_vmap=False, constrained=False) -> float:
        """
        At some point 'params' in parameter space, gets the hessian in unconstrained space and uses to estimate the
        model evidence
        :param params: Keyed dict with params in constrained / unconstrained parameter space
        :param data: data to condition the model on
        :param integrate_axes: Which axes to perform laplace approx for. If none, use all
        :param use_vmap: Whether to use jax vmapping over different params. (Not implemented)
        :param constrained: If true, perform laplace approx in constrained domain. Default to false
        :return: laplace log evidence as float or array of floats
        """

        self.msg_debug("-------------")
        self.msg_debug("Laplace Evidence eval")

        self.msg_debug("Constrained params are:", params)

        if integrate_axes is None:
            integrate_axes = self.paramnames()

        # Get 'height' and curvature of Gaussian
        if not constrained:
            uncon_params = self.to_uncon(params)

            self.msg_debug("Un-Constrained params are:", uncon_params)

            log_height = self.log_density_uncon(uncon_params, data)
            hess = self.log_density_uncon_hess(uncon_params, data, keys=integrate_axes)
        else:
            log_height = self.log_density(params, data)
            hess = self.log_density_hess(params, data, keys=integrate_axes)

        dethess = np.linalg.det(-hess)

        self.msg_debug("With determinant:", dethess)

        self.msg_debug("And log height: %.2f..." % log_height)

        D = len(integrate_axes)
        out = np.log(2 * np.pi) * (D / 2) - np.log(dethess) / 2 + log_height

        self.msg_debug("log-evidence is ~%.2f" % out)
        return out

    def laplace_log_info(self, params, data, integrate_axes=None, use_vmap=False, constrained=False):
        """
        At some point 'params' in parameter space, gets the hessian in unconstrained space and uses to estimate the
        model information relative to the prior
        :param params: site(s) to evaluate info at in the constrained domain
        :param data: data to condition the model on
        :param integrate_axes: free axes to perform laplace integral over. If none, use all
        :param use_vmap: Whether to use jax vmapping over different params. (Not implemented)
        :param constrained: If true perform laplace approx in the constrained domain
        :return: Laplace log info evaluated at params
        """

        if integrate_axes is None:
            integrate_axes = self.paramnames()

        if not constrained:
            uncon_params = self.to_uncon(params)

            log_height = self.log_density_uncon(uncon_params, data)
            hess = self.log_density_uncon_hess(uncon_params, data)
        else:
            log_height = self.log_density(params, data)
            hess = self.log_density_hess(params, data)

        I = np.where([key in integrate_axes for key in self.paramnames()])[0]

        hess = hess[I, I]
        if len(I) > 1:
            dethess = np.linalg.det(hess)
        else:
            dethess = hess

        # todo - double check sign on the log term. Might be wrong
        # todo - add case check for non-uniform priors.
        D = len(integrate_axes)
        out = -(np.log(2 * np.pi) + 1) * (D / 2) - np.log(-dethess) / 2 + np.log(self.prior_volume)
        return out

    def opt_tol(self, params: dict[str, float], data: _types.Any, integrate_axes: [str] = None, use_vmap: bool = False,
                constrained: bool = False) -> float:
        """
        Estimates the number of standard deviations away from the peak that a point is.
        Only works if you're nearby the local peak.

        :param params: site(s) to evaluate info at in the constrained domain
        :param data: data to condition the model on
        :param integrate_axes: free axes to perform laplace approx for. If none, use all
        :param use_vmap: Whether to use jax vmapping over different params. (Not implemented)
        :param constrained: If true, calculate distance in constrained domain approx in constrained domain. Default to false
        :return: Number of "standard deviations" away from local peak
        """
        if integrate_axes is None:
            integrate_axes = self.paramnames()

        # Get hessians and grads
        if not constrained:
            uncon_params = self.to_uncon(params)

            hess = self.log_density_uncon_hess(uncon_params, data, keys=integrate_axes)
            grad = self.log_density_uncon_grad(uncon_params, data, keys=integrate_axes)
        else:
            hess = self.log_density_hess(params, data, keys=integrate_axes)
            grad = self.log_density_grad(params, data, keys=integrate_axes)

        self.msg_debug("In .opt_tol, hessian is:\n", hess, "and grad is:\n", grad)

        # todo - remove this when properly integrating keys argument into grad funcs
        I = np.where([key in integrate_axes for key in grad.keys()])[0]
        grad = np.array([float(x) for x in grad.values()])[I]
        grad, hess = -grad, -hess

        # ------------------------------------------------
        # Calculate tolerances
        if np.linalg.det(hess) <= 0 or np.isnan(hess).any():
            if self.debug:
                eigs = np.linalg.eig(hess)[0]
                self.msg_debug("Eig vals are:", eigs, delim="\n\t")
            self.msg_err("In .opt_tol, Params appear to be diverged, broken or in saddle.", lvl=1)
            return np.inf

        try:
            Hinv = np.linalg.inv(hess)
            loss = np.dot(grad,
                          np.dot(
                              Hinv, grad
                          )
                          )
            return np.sqrt(abs(loss))

        except:
            self.msg_err("Undetermined error in .opt_tol. Returning inf.", lvl=1)
            return np.inf

    # --------------------------------
    # Sampling Utils
    def prior_sample(self, num_samples: int = 1, seed: int = None) -> dict:
        """
        Blind sampling from the prior without conditioning. Returns model parameters only
        :param num_samples: Number of realizations to generate
        :param seed: seed for random generation
        :return: keyed dict of parameters drawn from prior
        """

        if seed == None: seed = _utils.randint()

        pred = numpyro.infer.Predictive(self.prior,
                                        num_samples=num_samples,
                                        return_sites=self.paramnames()
                                        )

        params = pred(rng_key=jax.random.PRNGKey(seed))

        if num_samples == 1:
            params = {key: params[key][0] for key in params.keys()}
        return params

    def realization(self, data=None, num_samples: int = 1, seed: int = None):
        """
        Generates realizations of the observables by blindly sampling from the prior
        :param data: data to condition the lightcurve on
        :param num_samples: Number of realizations to generate
        :param seed: seed for random generation
        :return: keyed dict of the model observables
        """
        if seed == None: seed = _utils.randint()

        pred = numpyro.infer.Predictive(self.model_function,
                                        num_samples=num_samples,
                                        return_sites=None
                                        )

        params = pred(rng_key=jax.random.PRNGKey(seed), data=data)
        return params

    def _gen_lightcurve(self, data, params: dict, Tpred) -> (
            _types.ArrayN, _types.ArrayN, _types.ArrayNxN, _types.ArrayNxN):
        """
        At times Tpred and for parameters params and conditioned on some data, predict the signal mean and covariance.
        Returns like (loc_1, loc_2, covar_1,covar_2).
        This is a hidden function, to actually generate call in make_lightcurves(). This should be updated for a new model
        :param data: Data to condition the model on
        :param params: keyed dictionary of parameters
        :param Tpred: Array of time values to predict the lightcurve at
        :param num_samples: number of samples to draw from params use in integration to get covar / mu
        """

        loc_1 = np.zeros_like(Tpred)
        covar_1 = np.zeros([len(Tpred), len(Tpred)])
        loc_2, covar_2 = loc_1.copy(), covar_1.copy()

        return loc_1, loc_2, covar_1, covar_2

    def make_lightcurves(self, data, params: dict, Tpred, num_samples: int = 1) -> (lightcurve, lightcurve):
        """
        Returns lightcurves at time 'T' for 'parameters' conditioned on 'data' over `num_samples` draws from `params'

        :param data: Data to condition the model on
        :param params: keyed dictionary of parameters
        :param Tpred: Array of time values to predict the lightcurve at
        :param num_samples: number of samples to draw from params use in integration to get covar / mu
        :return: tuple of mean and covariance of LC's (loc_1, loc_2, covar_1, covar_2)
        """

        len_params = _utils.dict_dim(params)[1]
        if num_samples > len_params:
            self.msg_err("Warning! Tried to call %i samples from only %i parameters in make_lightcurves" % (
                num_samples, len_params),
                         lvl=1)

        loc_1 = np.zeros_like(Tpred)
        covar_1 = np.zeros([len(Tpred), len(Tpred)])
        loc_2, covar_2 = loc_1.copy(), covar_1.copy()

        if self._gen_lightcurve is stats_model._gen_lightcurve:
            self.msg_err("Warning, called make_lightcurves on a stats_model that doesn't have implementation", lvl=0)

        if not _utils.isiter_dict(params):
            loc_1, loc_2, covar_1, covar_2 = self.gen_lightcurve(data, params, jnp.array(Tpred))

        else:
            I = np.random.choice(range(len_params), num_samples, replace=True)
            loc_1_all = np.tile(loc_1, (num_samples, 1)) * 0
            loc_2_all = loc_1_all.copy()

            for k, p_sample in enumerate([_utils.dict_divide(params)[i] for i in I]):
                loc_1_i, loc_2_i, covar_1_i, covar_2_i = self.gen_lightcurve(data, p_sample, jnp.array(Tpred))
                covar_1 += covar_1_i
                covar_2 += covar_2_i
                loc_1_all[k, :] = loc_1_i
                loc_2_all[k, :] = loc_2_i
            loc_1 = np.mean(loc_1_all, axis=0)
            loc_2 = np.mean(loc_2_all, axis=0)
            covar_1 = covar_1 / num_samples + np.diag(np.var(loc_1_all, axis=0))
            covar_2 = covar_2 / num_samples + np.diag(np.var(loc_2_all, axis=0))

        err_1, err_2 = np.diag(covar_1) ** 0.5, np.diag(covar_2) ** 0.5

        outs = (lightcurve(Tpred, loc_1, err_1), lightcurve(Tpred, loc_2, err_2))

        return outs

    def params_inprior(self, params) -> bool:
        """
        Utility to check if model params fall within the uniform prior bounds
        :param params: constrained space params to check validity of
        :return: True if site falls within prior boundaries, false if not
        """

        isgood = {key: True for key in params.keys()}
        for key in params.keys():
            if key in self.fixed_params():
                if np.any(params[key] != self.prior_ranges[key][0]):
                    isgood[key] = False
                else:
                    isgood[key] = True
            else:
                if np.any(
                        not (
                                (params[key] >= self.prior_ranges[key][0]) * (params[key] < self.prior_ranges[key][1])
                        )
                ):
                    isgood[key] = False
                else:
                    isgood[key] = True
        return isgood

    def find_seed(self, data, guesses=None, fixed={}) -> (dict, float):
        """
        Find a good initial seed. Unless otherwise over-written, while blindly sample the prior and return the best fit.
        :param data: data to condition the model on
        :param guesses: number of evals to use for finding the seed
        :param fixed: keyed dict of parameters to be fixed instead of estimating a seed
        :return: tuple of dict of seed params and log density at this position
        """

        if len(fixed.keys()) == len(self.paramnames()): return fixed, self.log_density(fixed, data)

        if guesses == None: guesses = 50 * 2 ** len(self.free_params())

        samples = self.prior_sample(num_samples=guesses)

        if fixed != {}: samples = _utils.dict_extend(samples | fixed)

        ll = self.log_density(samples, data)
        i = ll.argmax()

        out = _utils.dict_divide(samples)[i]
        return out, ll.max()


# ============================================
# Custom statmodel example
class GP_simple(stats_model):
    """
    An example of how to construct your own stats_model in the simplest form.
    Requirements are to:
        1. Set a default prior range for all parameters used in model_function
        2. Define a numpyro generative model model_function
    You can add / adjust methods as required, but these are the only main steps
    """

    def __init__(self, prior_ranges=None, **kwargs):
        self._default_prior_ranges = {
            'lag': _default_config['lag'],
            'logtau': _default_config['logtau'],
            'logamp': _default_config['logamp'],
            'rel_amp': _default_config['rel_amp'],
            'mean': _default_config['mean'],
            'rel_mean': _default_config['rel_mean'],
        }
        self._protected_keys = ['basekernel']
        super().__init__(prior_ranges=prior_ranges, **kwargs)

        self.basekernel: tinygp.kernels.quasisep = kwargs[
            'basekernel'] if 'basekernel' in kwargs.keys() else tinygp.kernels.quasisep.Exp
        """The gaussian kernel"""

    # --------------------
    def prior(self) -> list[float, float, float, float, float, float]:
        # Sample distributions
        lag = quickprior(self, 'lag')

        logtau = quickprior(self, 'logtau')
        logamp = quickprior(self, 'logamp')

        rel_amp = quickprior(self, 'rel_amp')
        mean = quickprior(self, 'mean')
        rel_mean = quickprior(self, 'rel_mean')

        return lag, logtau, logamp, rel_amp, mean, rel_mean

    def model_function(self, data) -> None:
        lag, logtau, logamp, rel_amp, mean, rel_mean = self.prior()

        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag = jnp.square(E)

        delays = jnp.array([0, lag])
        amps = jnp.array([amp, rel_amp * amp])
        means = jnp.array([mean, mean + rel_mean])

        T_delayed = T - delays[bands]
        I = T_delayed.argsort()

        # Build and sample GP

        gp = gpw.build_gp(T_delayed[I], Y[I], diag[I], bands[I], tau, amps, means, basekernel=self.basekernel)
        numpyro.sample("Y", gp.numpyro_dist(), obs=Y[I])

    # -----------------------
    def _gen_lightcurve(self, data, params: dict, Tpred) -> (
            _types.ArrayN, _types.ArrayN, _types.ArrayNxN, _types.ArrayNxN):
        # Unpack params
        lag, logtau, logamp, rel_amp, mean, rel_mean = [params[key] for key in
                                                        ['lag', 'logtau', 'logamp', 'rel_amp', 'mean', 'rel_mean']]

        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]
        diag = jnp.square(E)

        # Conversions to gp-friendly form & making of matrices
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        delays = jnp.array([0, lag])
        amps = jnp.array([amp, rel_amp * amp])
        means = jnp.array([mean, mean + rel_mean])

        T_delayed = T - delays[bands]
        I = T_delayed.argsort()

        # Build and sample GP
        gp = gpw.build_gp(T_delayed[I], Y[I], diag[I], bands[I], tau, amps, means,
                          basekernel=self.basekernel)

        loc1, var1 = gp.predict(y=Y[I], X_test=(Tpred, jnp.zeros(len(Tpred), dtype=int)), return_cov=True)
        loc2, var2 = gp.predict(y=Y[I], X_test=(Tpred - lag, jnp.ones(len(Tpred), dtype=int)), return_cov=True)

        return loc1, loc2, var1, var2

    def find_seed(self, data, guesses=None, fixed={}) -> (float, dict[str, float]):

        # -------------------------
        # Setup
        T, Y, E, bands = [data[key] for key in ['T', 'Y', 'E', 'bands']]

        T1, Y1, E1 = T[bands == 0], Y[bands == 0], E[bands == 0]
        T2, Y2, E2 = T[bands == 1], Y[bands == 1], E[bands == 1]

        # If not specified, use roughly 4 per epoch of main lightcurve
        if guesses is None: guesses = int(np.array(self.prior_ranges['lag']).ptp() / np.median(np.diff(T1))) * 4

        check_fixed = self.params_inprior(fixed)
        if False in check_fixed:
            bad_keys = ['\t%s' % key for [key, val] in check_fixed.items() if val == False]
            self.msg_err("Warning! Tried to fix seed params at values that lie outside of prior range:", *bad_keys,
                         delim="\n", lvl=0)

        # -------------------------
        # Estimate Correlation Timescale
        if 'logtau' not in fixed.keys():
            from litmus.ICCF_working import correlfunc_jax_vmapped
            approx_season = np.diff(T1).max()

            if approx_season > np.median(np.diff(T1)) * 5:
                span = approx_season
            else:
                span = np.ptp(T1) * 0.1

            autolags = jnp.linspace(-span, span, 1024)
            autocorrel = correlfunc_jax_vmapped(autolags, T1, Y1, T1, Y1, 1024)

            autolags, autocorrel = np.array(autolags), np.array(autocorrel)

            # Trim to positive values and take a linear regression
            autolags = autolags[autocorrel > 0]
            autocorrel = autocorrel[autocorrel > 0]
            autocorrel = np.log(autocorrel)
            autocorrel[autolags < 0] *= -1

            autolags -= autolags.mean(),
            autocorrel -= autocorrel.mean()

            tau = (autolags * autolags).sum() / (autolags * autocorrel).sum()
            tau = abs(tau)
            # tau *= np.exp(1)
        else:
            tau = 1

        # -------------------------
        # Estimate mean & variances
        Y1bar, Y2bar = np.average(Y1, weights=E1 ** -2), np.average(Y2, weights=E2 ** -2)
        Y1var, Y2var = np.average((Y1 - Y1bar) ** 2, weights=E1 ** -2), np.average((Y2 - Y2bar) ** 2, weights=E2 ** -2)

        # -------------------------

        out = {
            'lag': 0.0,
            'logtau': np.log(tau),
            'logamp': np.log(Y1var) / 2,
            'rel_amp': np.sqrt(Y2var / Y1var),
            'mean': Y1bar,
            'rel_mean': Y2bar - Y1bar,
        }

        out |= fixed

        # -------------------------
        # For any extended models, randomly sample
        extended_keys = [key for key in self.free_params() if key not in out.keys()]
        if len(extended_keys) != 0:
            self.msg_debug("In GP_simple.find_seed(), using blind sampeling to find guesses for ", *extended_keys)
            extended_samps = _utils.dict_extend(self.prior_sample(guesses) | out)
            extended_lls = self.log_density(extended_samps, data)
            out = _utils.dict_divide(extended_samps)[np.argmax(extended_keys)]

        # -------------------------
        # Where estimates are outside prior range, round down
        isgood = self.params_inprior(out)
        r = 0.01
        isgood['lag'] = True
        for key in out.keys():
            if not isgood[key]:
                if out[key] < self.prior_ranges[key][0]:
                    out[key] = self.prior_ranges[key][0] + r * np.ptp(self.prior_ranges[key])
                else:
                    out[key] = self.prior_ranges[key][1] - r * np.ptp(self.prior_ranges[key])

        # -------------------------
        # Estimate lag with a sweep if not in fixed
        if 'lag' not in fixed.keys():
            lag_fits = np.linspace(*self.prior_ranges['lag'], guesses, endpoint=False)
            out |= {'lag': lag_fits}
        else:
            out |= {'lag': fixed['lag']}

        # -------------------------
        # Get log likelihoods and return best value
        out = _utils.dict_extend(out)

        lls = self.log_density(params=out, data=data)
        if _utils.dict_dim(out)[1] > 1:
            i = lls.argmax()
            ll_out = lls[i]
            out = {key: out[key][i] for key in out.keys()}
            self.msg_debug("In find seed, sample no %i is best /w LL %.2f at lag %.2f" % (i, ll_out, out['lag']))
        else:
            ll_out = float(lls)

        return out, ll_out


# ------------------
class GP_simple_null(GP_simple):
    """
    A variant of GP_simple for uncoupled gaussian processes, equivalent to lag->infty in GP_simple.
    Used for null hypothesis testing through model comparison.
    """

    def __init__(self, prior_ranges=None, **kwargs):
        self._default_prior_ranges = {
            'lag': [0.0, 0.0]
        }
        super().__init__()

    def lc_to_data(self, lc_1: lightcurve, lc_2: lightcurve) -> dict:
        return super().lc_to_data(lc_1, lc_2) | {
            'T1': lc_1.T,
            'Y1': lc_1.Y,
            'E1': lc_1.E,
            'T2': lc_2.T,
            'Y2': lc_2.Y,
            'E2': lc_2.E,
        }

    def model_function(self, data) -> None:
        lag, logtau, logamp, rel_amp, mean, rel_mean = self.prior()

        T1, Y1, E1 = [data[key] for key in ['T1', 'Y1', 'E1']]
        T2, Y2, E2 = [data[key] for key in ['T2', 'Y2', 'E2']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag1, diag2 = jnp.square(E1), jnp.square(E2)

        # amps = jnp.array([amp, rel_amp * amp])
        # means = jnp.array([mean, mean + rel_mean])
        Y1 -= mean
        Y2 -= (mean + rel_mean)

        # Build and sample GP
        kernel1 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp)
        kernel2 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp * rel_amp)

        gp1 = GaussianProcess(kernel1, T1, diag=diag1)
        gp2 = GaussianProcess(kernel2, T2, diag=diag2)
        numpyro.sample("Y1", gp1.numpyro_dist(), obs=Y1)
        numpyro.sample("Y2", gp2.numpyro_dist(), obs=Y2)

    def _gen_lightcurve(self, data, params: dict, Tpred) -> (
            _types.ArrayN, _types.ArrayN, _types.ArrayNxN, _types.ArrayNxN):
        # Unpack params
        logtau, logamp, rel_amp, mean, rel_mean = [params[key] for key in
                                                   ['logtau', 'logamp', 'rel_amp', 'mean', 'rel_mean']]

        T1, Y1, E1 = [data[key] for key in ['T1', 'Y1', 'E1']]
        T2, Y2, E2 = [data[key] for key in ['T2', 'Y2', 'E2']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag1, diag2 = jnp.square(E1), jnp.square(E2)

        # amps = jnp.array([amp, rel_amp * amp])
        # means = jnp.array([mean, mean + rel_mean])
        Y1 -= mean
        Y2 -= (mean + rel_mean)

        # Build and sample GP
        kernel1 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp)
        kernel2 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp * rel_amp)

        gp1 = GaussianProcess(kernel1, T1, diag=diag1)
        gp2 = GaussianProcess(kernel2, T2, diag=diag2)

        loc1, var1 = gp1.predict(y=Y1, X_test=Tpred, return_cov=True)
        loc2, var2 = gp2.predict(y=Y2, X_test=Tpred, return_cov=True)

        return loc1, loc2, var1, var2


class whitenoise_null(GP_simple_null):

    def model_function(self, data) -> None:
        lag, logtau, logamp, rel_amp, mean, rel_mean = self.prior()

        T1, Y1, E1 = [data[key] for key in ['T1', 'Y1', 'E1']]
        T2, Y2, E2 = [data[key] for key in ['T2', 'Y2', 'E2']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag1, diag2 = jnp.square(E1), jnp.square(E2)

        # amps = jnp.array([amp, rel_amp * amp])
        # means = jnp.array([mean, mean + rel_mean])
        Y1 -= mean

        # Build and sample GP
        kernel1 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp)

        gp1 = GaussianProcess(kernel1, T1, diag=diag1)
        numpyro.sample("Y1", gp1.numpyro_dist(), obs=Y1)

        with numpyro.plate("whitenoise", len(Y2)):
            numpyro.sample('Y2',
                           dist.Normal(mean + rel_mean, jnp.sqrt((amp * rel_amp) ** 2 + E2 ** 2)),
                           obs=Y2)

    def _gen_lightcurve(self, data, params: dict, Tpred) -> (
            _types.ArrayN, _types.ArrayN, _types.ArrayNxN, _types.ArrayNxN):
        # Unpack params
        logtau, logamp, rel_amp, mean, rel_mean = [params[key] for key in
                                                   ['logtau', 'logamp', 'rel_amp', 'mean', 'rel_mean']]

        T1, Y1, E1 = [data[key] for key in ['T1', 'Y1', 'E1']]
        T2, Y2, E2 = [data[key] for key in ['T2', 'Y2', 'E2']]

        # Conversions to gp-friendly form
        amp, tau = jnp.exp(logamp), jnp.exp(logtau)

        diag1, diag2 = jnp.square(E1), jnp.square(E2)

        # amps = jnp.array([amp, rel_amp * amp])
        # means = jnp.array([mean, mean + rel_mean])
        Y1 -= mean
        Y2 -= (mean + rel_mean)

        # Build and sample GP
        kernel1 = tinygp.kernels.quasisep.Exp(scale=tau, sigma=amp)

        gp1 = GaussianProcess(kernel1, T1, diag=diag1)

        loc1, var1 = gp1.predict(y=Y1, X_test=Tpred, return_cov=True)
        loc2, var2 = np.ones_like(Tpred) * (mean + rel_mean), np.ones_like(Tpred) * (amp * rel_amp) ** 2

        return loc1, loc2, var1, var2


# ================================================
# ================================================

class GP_simple_normalprior(GP_simple):
    def __init__(self, prior_ranges=None, **kwargs):
        self._default_prior_ranges = {
            'lag': [1.0, 1000.0]
        }
        self._protected_keys = ['mu_lagpred', 'sig_lagpred']
        super().__init__()

        # Default values for hbeta AGN @ ~44 dex lum, from R-L relations
        self.mu_lagpred = 1.44 * np.log(10)  # ~28 days, from McDougall et al 2025a
        self.sig_lagpred = np.log(10) * 0.24  # ~1.75 dex, from McDougall et al 2025a

    def prior(self) -> (float, float, float, float, float, float):

        tform = numpyro.distributions.transforms.ExpTransform()
        tformed_dist = numpyro.distributions.TransformedDistribution(
            dist.TruncatedNormal(self.mu_lagpred, self.sig_lagpred, low=jnp.log(self.prior_ranges['lag'][0]),
                                 high=jnp.log(self.prior_ranges['lag'][1])), [tform, ]
        )
        lag = numpyro.sample('lag', tformed_dist)
        masked_model = numpyro.handlers.substitute(super().prior, {'lag': lag})
        with numpyro.handlers.block(hide=['lag']):
            _, logtau, logamp, rel_amp, mean, rel_mean = masked_model()
        return lag, logtau, logamp, rel_amp, mean, rel_mean

    def uncon_grad_lag(self, params) -> float:

        from numpyro.infer.util import transform_fn

        if 'lag' not in self.paramnames(): return 0
        if np.ptp(self.prior_ranges['lag']) == 0: return 0

        tform = numpyro.distributions.transforms.ExpTransform()
        lagdist = numpyro.distributions.TransformedDistribution(
            dist.TruncatedNormal(self.mu_lagpred, self.sig_lagpred, low=jnp.log(self.prior_ranges['lag'][0]),
                                 high=jnp.log(self.prior_ranges['lag'][1])), [tform, ]
        )
        lag_con = params['lag']

        transforms = {"lag": numpyro.distributions.biject_to(lagdist.support)}

        def tform(x):
            out = transform_fn(transforms, params | {'lag': x}, invert=True)['lag']
            return out

        tform = jax.grad(tform)
        out = np.log(abs(tform(lag_con)))
        return out
