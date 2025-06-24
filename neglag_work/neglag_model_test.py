from neglag_model import model_twolag

import matplotlib.pyplot as plt
import jax
import numpy as np

import litmus.models
from litmus import *
import jax.numpy as jnp
import numpyro



mock = litmus.mocks.mock(season=None,
                         cadence = [7,30],)
mock.plot()
model = model_twolag(prior_ranges={
    'rel_amp': 1.0,
    'logamp': 0.0,
    'logtau': np.log(mock.tau),
    'mean': 0.0,
    'rel_mean': 0.0,
    'lag_two': [-100,0],
    'lag_relamp': [0,0.5],
    'lag': [0,100]
})
data = model.lc_to_data(mock.lc_1, mock.lc_2)

params = model.prior_sample() | {
    'lag': 30, 'lag_two': -100, 'lag_relamp':0.5
}
test_covar = model.make_matrix(params, data)

I = np.argsort(data['bands'])
plt.imshow(test_covar[I,:][:,I])
plt.show()

test_params = model.prior_sample(32)
model.log_density(test_params, data)

# ----------------------
fitter = litmus.fitting_methods.JAVELIKE(model, verbose=5, debug=5,
                                                num_parallel_samplers=8,
                                                num_live_points=150)
fitter.fit(mock.lc_1, mock.lc_2)
lt = litmus.LITMUS(fitter)
lt.diagnostic_plots()
lt.plot_parameters(prior_extents=False)