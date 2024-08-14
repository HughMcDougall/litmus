from pypolychord import run
from litmus.mocks import mock_C
import jax

mock = mock_C()

#import matplotlib.pyplot as plt
#plt.plot(mock.lc.T, mock.lc.Y,'.')
#plt.plot(mock.lc_1.T, mock.lc_1.Y,'-')
#plt.plot(mock.lc_2.T, mock.lc_2.Y,'-')

from litmus.models import GP_simple
test_model = GP_simple()
params = test_model.prior_sample()
data = test_model.lc_to_data(mock.lc_1, mock.lc_2)
params
test_model._log_density(params, data)

prior_ranges = test_model.prior_ranges

import numpy as np
mn, mx = np.array([v for v in prior_ranges.values()]).T
from pypolychord.priors import UniformPrior
prior = UniformPrior(mn, mx)
nDims = len(params)
theta = prior(np.random.rand(nDims))

@jax.jit
def jax_loglikelihood(theta):
    p = {k:v for k,v in zip(prior_ranges.keys(), theta)}
    return test_model._log_density(p, data)

def loglikelihood(theta):
    return jax_loglikelihood(theta).item()

import jax.numpy as jnp 

# un-comment to generate yourself
#samples = run(loglikelihood, nDims, prior=prior)

# Load samples from will handley's run
from anesthetic import read_chains
samples = read_chains('test.csv')

samples.logZ() #log-evidence
samples.logZ(1000).std() #error on log-evidence

samples.gui() # replaying of the run


