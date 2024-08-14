'''
Test code for the fitting_methods module

HM 14/8/24
'''

#===============================

from litmus.fitting_methods import *
#===============================


from mocks import mock_A, mock_B, mock_C

import matplotlib.pyplot as plt
from models import dummy_statmodel

#::::::::::::::::::::

mock = mock_A(seed=5)
mock01 = mock.lc_1
mock02 = mock.lc_2
lag_true = mock.lag

plt.figure()
mock().plot(axis=plt.gca())
plt.legend()
plt.grid()
plt.show()

#::::::::::::::::::::
test_statmodel = dummy_statmodel()

# ICCF Test
test_ICCF = ICCF(Nboot=128, Nterp=128, Nlags=128, stat_model=test_statmodel)
print("Doing Fit")
test_ICCF.fit(mock01, mock02)
print("Fit done")

ICCF_samples = test_ICCF.get_samples()['lag']
print(ICCF_samples.mean(), ICCF_samples.std())

plt.figure()
plt.hist(ICCF_samples, histtype='step', bins=24)
plt.axvline(lag_true, ls='--', c='k', label="True Lag")
plt.axvline(ICCF_samples.mean(), ls='--', c='r', label="Mean Lag")
plt.title("ICCF Results")
plt.legend()
plt.grid()
plt.show()

# ---------------------
# Prior Sampling

test_prior_sampler = prior_sampling(stat_model=test_statmodel)
test_prior_sampler.fit(lc_1=mock01, lc_2=mock02, seed=0)
test_samples = test_prior_sampler.get_samples(512, importance_sampling=True)

plt.figure()
plt.title("Dummy prior sampling test")
plt.hist(test_samples['lag'], histtype='step', density=True)
plt.axvline(250.0, ls='--', c='k')
plt.axvline(test_samples['lag'].mean(), ls='--', c='r')
plt.ylabel("Posterior Density")
plt.xlabel("Lag")
plt.grid()
plt.tight_layout()
plt.show()
