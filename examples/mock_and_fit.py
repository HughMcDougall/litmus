"""
An example of how to generate a mock a'la the OzDES sample and generate a
"""

import numpy as np
from litmus_rm.mocks import mock
from litmus_rm.fitting_methods import hessian_scan, JAVELIKE
from litmus_rm.models import GP_simple, GP_simple_null

seed = 1
np.random.seed(seed)

# ------------------------
loglag_mu, loglag_sig = 1.996773134, 0.405083319
lag = np.power(10, np.random.randn() * loglag_sig + loglag_mu)
logtau = np.log(10)

print("Randomly selected True lag is %.2f" % lag)

mymock = mock(seed=seed, lag=lag, tau=np.exp(logtau), cadence=30)
mymock.plot()

# ------------------------

model_alt = GP_simple()
model_alt.set_priors({'rel_mean': 0.0, 'rel_amp': 1.0, 'logamp': 0.0, 'mean':0.0})
model_null = GP_simple_null()

fitter_alt = hessian_scan(model_alt, verbose=10, Nlags=32, warnings=10, debug=10)
fitter_null = hessian_scan(model_null)

# Fitting

fitter_null.fit(*mymock.lcs())
fitter_alt.fit(*mymock.lcs())

try:
    print(fitter_null.get_evidence())
except:
    print("FAIL ON NULL")

try:
    print(fitter_alt.get_evidence())
except:
    print("FAIL ON NULL")