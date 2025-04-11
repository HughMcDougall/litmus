from litmus.fitting_methods import *

# =====================================================
if __name__ == "__main__":
    import mocks

    import matplotlib.pyplot as plt
    from litmus.models import dummy_statmodel

    #::::::::::::::::::::

    mock = mocks.mock(seed=2, lag=100, season=0)
    mock01 = mock.lc_1
    mock02 = mock.lc_2
    lag_true = mock.lag

    mock().plot()
    #::::::::::::::::::::
    test_statmodel = dummy_statmodel()

    # ICCF Test
    print("Doing ICCF Testing:")
    test_ICCF = ICCF(Nboot=128, Nterp=1024, Nlags=1024, stat_model=test_statmodel, debug=True)
    print("\tDoing Fit")
    test_ICCF.fit(mock01, mock02)
    print("\tFit done")

    ICCF_samples = test_ICCF.get_samples()['lag']
    print("\tICCF Results , dT = %.2f +/- %.2f compared to true lag of %.2f"
          % (ICCF_samples.mean(), ICCF_samples.std(), lag_true)
          )

    plt.figure()
    plt.hist(ICCF_samples, histtype='step', bins=24)
    plt.axvline(lag_true, ls='--', c='k', label="True Lag")
    plt.axvline(ICCF_samples.mean(), ls='--', c='r', label="Mean Lag")
    plt.axvline(ICCF_samples.mean() + ICCF_samples.std(), ls=':', c='r')
    plt.axvline(ICCF_samples.mean() - ICCF_samples.std(), ls=':', c='r')
    plt.title("ICCF Results")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------
    # Prior Sampling

    print("Doing PriorSampling Testing:")
    test_prior_sampler = prior_sampling(stat_model=test_statmodel)
    print("\tDoing Fit")
    test_prior_sampler.fit(lc_1=mock01, lc_2=mock02, seed=0)
    print("\tFit done")

    test_samples = test_prior_sampler.get_samples(512, importance_sampling=True)
    print("\tPriorSampling Results , dT = %.2f +/- %.2f compared to true lag of %.2f"
          % (test_samples['lag'].mean(), test_samples['lag'].std(), test_statmodel.lag_peak)
          )

    plt.figure()
    plt.title("Prior Sampling")
    plt.hist(test_samples['lag'], histtype='step', density=True)
    plt.axvline(test_statmodel.lag_peak, ls='--', c='k')
    plt.axvline(test_samples['lag'].mean(), ls='--', c='r')
    plt.axvline(test_samples['lag'].mean() + test_samples['lag'].std(), ls=':', c='r')
    plt.axvline(test_samples['lag'].mean() - test_samples['lag'].std(), ls=':', c='r')
    plt.ylabel("Posterior Density")
    plt.xlabel("Lag")
    plt.grid()
    plt.tight_layout()
    plt.show()
