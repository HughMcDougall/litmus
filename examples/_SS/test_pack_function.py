'''
An example and test of how to optimize backed functions
'''

# ===============================
import numpy as np
import matplotlib.pyplot as plt

import jaxopt

import models
from _utils import *

import numpyro

import jax.numpy as jnp


# ===============================


def model():
    x = numpyro.sample('x', numpyro.distributions.Normal(0, 1))
    y = numpyro.sample('y', numpyro.distributions.Normal(10, 2))
    z = numpyro.sample('z', numpyro.distributions.Normal(-5, 3))


def f(params):
    out = numpyro.infer.util.log_density(model, (), {}, params)[0]
    return (out)


def f_uncon(params):
    out = -1 * numpyro.infer.util.potential_energy(model, (), {}, params)
    return (out)


# ---------------------------------------------------------------
packed_keys = ['z', 'y']
truth = {
    'x': 0.0,
    'y': 10.0,
    'z': -5.0,
}

f_pack = pack_function(f, packed_keys=packed_keys,
                       fixed_values={key: truth[key] for key in truth.keys() if key not in packed_keys}, invert=False)
f_pack_invert = pack_function(f, packed_keys=packed_keys,
                              fixed_values={key: truth[key] for key in truth.keys() if key not in packed_keys},
                              invert=True)
f_map = jax.vmap(jax.vmap(f_pack, 0, 0), 1, 1)
f_map = jax.jit(f_map)

X, Y = np.linspace(-20, 20, 1024), np.linspace(-20, 20, 1024)
X, Y = np.meshgrid(X, Y)
plt.imshow(np.exp(np.flip(f_map([X, Y]), axis=0)), extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.axvline(truth[packed_keys[0]], ls='--', c='w', alpha=0.5, lw=4)
plt.axhline(truth[packed_keys[1]], ls='--', c='w', alpha=0.5, lw=4)

plt.xlabel(packed_keys[0]), plt.ylabel(packed_keys[1])

# ---------------------------------------------------------------

start_params = {
    'x': 0.0,
    'y': 0.0,
    'z': 0.0,
}

# ---------------------------------------------------------------
# Optimization with raw jaxopt
opt = jaxopt.GradientDescent(fun=f_pack_invert, stepsize=0.01, maxiter=1_000)
res, _ = opt.run(init_params=[start_params[key] for key in packed_keys])

print(res)

plt.scatter(*[start_params[key] for key in packed_keys], c='b')
plt.scatter(*res, c='r')
plt.show()

# ---------------------------------------------------------------
# Optimization with jaxopt tick
opt2 = jaxopt.GradientDescent(fun=f_pack_invert, stepsize=0.01, maxiter=1_000)
x0 = jnp.array([start_params[key] for key in packed_keys])
init_state = opt2.init_state(init_params=x0)

p, s = x0, init_state
for i in range(1_000):
    A = opt2.update(params=p, state=s)
    p, s = A.params, A.state

    if i % 100 == 0: print(i, p)


# ---------------------------------------------------------------
# Optimization with scan
class test_model(models.stats_model):

    def __init__(self):
        self._default_prior_ranges = {'x': [-100, 100],
                                      'y': [-100, 100],
                                      'z': [-100, 100]}

        super().__init__()

    def prior(self):
        x = models.quickprior(self, 'x')
        y = models.quickprior(self, 'y')
        z = models.quickprior(self, 'z')

        return (x, y, z)

    def model_function(self, data):
        x, y, z = self.prior()

        numpyro.sample('data_x', numpyro.distributions.Normal(0, 1), obs=x)
        numpyro.sample('data_y', numpyro.distributions.Normal(10, 2), obs=y)
        numpyro.sample('data_z', numpyro.distributions.Normal(-5, 3), obs=z)


test = test_model()
from mocks import mock_B

data = test.lc_to_data(mock_B.lc_1, mock_B.lc_2)
final_params = test.scan(start_params=start_params, data=data, stepsize=0.001)

print(final_params)
# ---------------------------------------------------------------
# Optimization with jaxopt tick of unconstrained domain
f_pack_uncon = pack_function(f_uncon, packed_keys=packed_keys,
                             fixed_values={key: truth[key] for key in truth.keys() if key not in packed_keys},
                             invert=False)
f_pack_incon_invert = pack_function(f_uncon, packed_keys=packed_keys,
                                    fixed_values={key: truth[key] for key in truth.keys() if key not in packed_keys},
                                    invert=True)
start_params_uncon = test.to_uncon(start_params)

opt3 = jaxopt.GradientDescent(fun=f_pack_incon_invert, stepsize=0.01, maxiter=1_000)
x0 = jnp.array([start_params_uncon[key] for key in packed_keys])
init_state = opt3.init_state(init_params=x0)

p, s = x0, init_state
for i in range(1_000):
    A = opt3.update(params=p, state=s)
    p, s = A.params, A.state
    if i % 100 == 0: print(i, p)


plt.plot(test.log_density_uncon(params=test.to_uncon(dict_extend(start_params | {'x': np.linspace(-20,20,128)})), data=data))