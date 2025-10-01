'''
A test case of applying polychord, courtesy of Will Handley https://github.com/williamjameshandley
Edited HM 14/8/24
'''

# =======================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use('TkAgg')

import jax
import jax.numpy as jnp

from pypolychord import run
from pypolychord.priors import UniformPrior
from anesthetic import read_chains

from chainconsumer import ChainConsumer

from litmus_rm.mocks import mock, mock_A, mock_B, mock_C
from litmus_rm.models import GP_simple

# =======================================

for itter, mymock in enumerate([
    mock(cadence=[7, 30], E=[0.05, 0.2], season=None, lag=100, tau=400.0),
    mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=100, tau=400.0),
    mock(cadence=[7, 30], E=[0.05, 0.2], season=180, lag=180, tau=200.0),
    mock(cadence=[7, 30], E=[0.1, 0.4], season=180, lag=180, tau=200.0),
    ]):

    mymock(itter)

    if True:
        plt.figure()
        ax = plt.gca()
        mymock.plot(axis=ax, series_args={'lw': [1, 1]})
        plt.grid()
        ax.set_xlim(0, mock_B.maxtime)
        plt.gcf().supxlabel("Signal Time (Days)")
        plt.title("Signal %i" %itter)
        plt.show()

    # Create a model, choose a prior location for testing, wrap data into correct format
    test_model = GP_simple()
    data = test_model.lc_to_data(mymock.lc_1, mymock.lc_2)
    params = test_model.prior_sample()

    test_model.set_priors({'lag': [mymock.lag, mymock.lag]})

    print("A param position:")
    for x in params.items():
        a, b = x
        print('\t %s \t %.2f' % (a, b))
    print("ll is %.2e" % test_model.log_density(params, data))

    prior_ranges = test_model.prior_ranges

    # ---------
    # POLYCHORD SETUP

    mn, mx = np.array([v for v in prior_ranges.values()]).T
    #  In PC this is a callable that converts from the unit uniform domain to the true prior space
    prior = UniformPrior(mn, mx)
    nDims = len(params)
    theta = prior(np.random.rand(nDims))  # Start loc for testing


    # PC takes the log-likelihood in _unconstrained_ prior space
    def loglikelihood(theta):
        p = {k: v for k, v in zip(prior_ranges.keys(), theta)}
        return float(test_model.log_density(p, data))


    print("To confirm, in function that PC relies on...")
    print("ll is %.2e" % loglikelihood(theta))

    # ---------
    # RUN POLYCHORD

    # If true re-run from scratch, else draw in from an existing run
    paramnames = [(a, b) for a, b in zip(['p%i' % i for i in range(nDims)], test_model.paramnames())]
    if True:
        nlive = 50 * (len(test_model.free_params()) + 1) * 2
        print("I am running with %i live points" % nlive)
        samples = run(loglikelihood, nDims, nlive=nlive, prior=prior, read_resume=False,
                      write_resume=False, paramnames=paramnames)
        samples.to_csv("polychord_test_%i.csv"%itter)
        print("Sampling Done!")
    else:
        samples = read_chains('run_samples.csv')

        # ---------
        # EXTRACT FUN STUFF USING ANESTHETIC

        lZ = samples.logZ()  # log-evidence
        lZ_d = samples.logZ(1000).std()  # error on log-evidence

        print("Log evidence is %.2f +/- %.2f" % (lZ, lZ_d))

        samples.gui()  # replaying of the run

        cc_samps = {key: val for key, val in zip(test_model.par1amnames(), samples.posterior_points().to_numpy().T) if
                    val.ptp() > 0}
        C = ChainConsumer()
        C.add_chain(cc_samps)
        C.plotter.plot(truth=mymock.params())

        plt.show()
