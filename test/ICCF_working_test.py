# =================================================
# Testing

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from litmus_rm.ICCF_working import *

    #::::::::::::::::::::
    # Mock Data
    from litmus_rm.mocks import mock_A, mock_B, mock_C

    mock = mock_A
    mock01 = mock.lc_1
    mock02 = mock.lc_2

    X1, Y1, E1 = mock01.T, mock01.Y, mock01.E
    X2, Y2, E2 = mock02.T, mock02.Y, mock02.E

    true_lag = mock.lag

    #::::::::::::::::::::
    # Plot Mock Signals

    plt.figure()
    plt.errorbar(X1, Y1, E1, fmt='none', capsize=2)
    plt.errorbar(X2, Y2, E2, fmt='none', capsize=2, c='tab:orange')
    plt.grid()
    plt.legend(["Main Signal", "Delayed Signal"])
    plt.show()

    #::::::::::::::::::::
    # Do JITted ICCF fit
    Nlags = 512
    Nterp, Nboot = 1024, 1024
    print("Running Fit")
    lags = np.linspace(-X1.ptp() / 2, X1.ptp() / 2, Nlags)
    correls_jax = correlfunc_jax_vmapped(lags, X1, Y1, X2, Y2, Nterp)
    jax_samples = correl_func_boot_jax_wrapper_nomap(lags, X1, Y1, X2, Y2, E1, E2, Nterp=Nterp, Nboot=Nboot)

    #::::::::::::::::::::
    print("ICCF RESULTS:")
    res_mean, res_std = jax_samples.mean(), jax_samples.std()
    res_p1, res_med, res_p2 = [np.percentile(jax_samples, p) for p in [14, 50, 86]]
    if res_med > true_lag:
        z_med = (res_med - true_lag) / (res_p2 - res_med)
    else:
        z_med = (true_lag - res_med) / (res_med - res_p1)

    print("Mean Statistics:")
    print("Lag = %.2f +/- %.2f, consistent with true lag of %.2f at %.2f sigma" % (
        res_mean, res_std, true_lag, abs(true_lag - res_mean) / res_std))
    print("Median Statistics:")
    print("Lag = %.2f + %.2f, -%.2f, consistent with true lag of %.2f at %.2f sigma" % (
        res_med, res_p2 - res_med, res_med - res_p1, true_lag, z_med))

    #::::::::::::::::::::
    # Plot Results
    range = np.array([-true_lag, true_lag * 3])

    plt.figure()
    plt.plot(lags, correls_jax / range.ptp() * 4, label="Un-Bootstrapped ICCF Correlation")
    plt.axhline(0, c='k')

    plt.hist(jax_samples, density=True, bins=32, alpha=0.75, label="ICCF Samples", range=range)

    plt.axvline(true_lag, c='k', ls='--', label="True Lag")

    plt.axvline(res_mean, c='tab:red', ls='--', label="ICCF Mean $\pm$ std")
    plt.axvline(res_mean - res_std, c='tab:red', ls=':')
    plt.axvline(res_mean + res_std, c='tab:red', ls=':')

    plt.axvline(res_med, c='tab:blue', ls='--', label="ICCF Med $\pm 1 \sigma$")
    plt.axvline(res_p1, c='tab:blue', ls=':')
    plt.axvline(res_p2, c='tab:blue', ls=':')

    plt.ylim(0, 1.5 / range.ptp() * 6)
    plt.xlim(*range)
    plt.xlabel("Lag")
    plt.ylabel("ICCF Correl (Normalized")
    plt.legend(loc='upper right')
    plt.grid()

    plt.show()
