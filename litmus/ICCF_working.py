'''
ICCF_working.py

JAX-friendly working for performing the ICCF fit. To be called by a fitting method

'''

import jax
import jax.numpy as jnp

#=================================================

def correl_jax(X1, Y1, X2, Y2, Nterp=1024):
    Xmin, Xmax = [f(jnp.array([f(X1), f(X2)])) for f in (jnp.min, jnp.max)]
    X_interp = jnp.linspace(Xmin, Xmax, 1024)
    Y1_interp = jnp.interp(X_interp, X1, fp=Y1, left=0, right=0)
    Y2_interp = jnp.interp(X_interp, X2, fp=Y2, left=0, right=0)
    out = jnp.corrcoef(x=Y1_interp, y=Y2_interp)[0][1]
    return (out)


correl_jax_jitted = jax.jit(correl_jax, static_argnames=["Nterp"])


#::::::::

def correlfunc_jax(lag, X1, Y1, X2, Y2, Nterp=1024):
    return (
        correl_jax(X1 - lag, Y1, X2, Y2, Nterp)
    )


correlfunc_jax_vmapped = jax.vmap(correlfunc_jax, in_axes=(0, None, None, None, None, None))


#::::::::

def correl_func_boot_jax(seed, lags, X1, Y1, X2, Y2, E1, E2, Nterp=1024, N1=2, N2=2):
    key = jax.random.key(seed)

    I1 = jax.random.choice(key=key, a=jnp.arange(X1.size), shape=(N1,), replace=False)
    I2 = jax.random.choice(key=key, a=jnp.arange(X2.size), shape=(N2,), replace=False)
    I1, I2 = jnp.sort(I1), jnp.sort(I2)

    X1p, X2p = X1[I1], X2[I2]
    Y1p = Y1[I1] + jax.random.normal(key, shape=(I1.size,)) * E1[I1]
    Y2p = Y2[I2] + jax.random.normal(key, shape=(I2.size,)) * E2[I2]
    correls = correlfunc_jax_vmapped(lags, X1p, Y1p, X2p, Y2p, 1024)
    i_max = jnp.argmax(correls)
    return (lags[i_max])


correl_func_boot_jax_nomap = jax.jit(correl_func_boot_jax, static_argnames=["Nterp", "N1", "N2"])

correl_func_boot_jax = jax.vmap(correl_func_boot_jax,
                                in_axes=(0, None, None, None, None, None, None, None, None, None, None))
correl_func_boot_jax = jax.jit(correl_func_boot_jax, static_argnames=["Nterp", "N1", "N2"])


def correl_func_boot_jax_wrapper(lags, X1, Y1, X2, Y2, E1, E2, Nterp=1024, Nboot=512, r=jnp.exp(-1)):
    seeds = jnp.arange(Nboot)
    N1, N2 = int(len(X1) * r), int(len(X2) * r)

    out = correl_func_boot_jax(seeds, lags, X1, Y1, X2, Y2, E1, E2, Nterp, N1, N2)

    return (out)


def correl_func_boot_jax_wrapper_nomap(lags, X1, Y1, X2, Y2, E1, E2, Nterp=1024, Nboot=512, r=jnp.exp(-1)):
    seeds = jnp.arange(Nboot)
    N1, N2 = int(len(X1) * r), int(len(X2) * r)

    out = [correl_func_boot_jax_nomap(seed, lags, X1, Y1, X2, Y2, E1, E2, Nterp, N1, N2) for seed in range(Nboot)]

    return (jnp.array(out))


#=================================================
# Testing

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    #::::::::::::::::::::
    # Mock Data
    print("Making Mock Data")
    f = lambda x: np.exp(-((x - 8) / 2) ** 2 / 2)

    X1 = np.linspace(0, 2 * np.pi * 3, 64)
    X2 = np.copy(X1)[::2]
    X1 += np.random.randn(len(X1)) * X1.ptp() / (len(X1) - 1) * 0.25
    X2 += np.random.randn(len(X2)) * X2.ptp() / (len(X2) - 1) * 0.25

    E1, E2 = [np.random.poisson(10, size=len(X)) * 0.005 for i, X in enumerate([X1, X2])]
    E2 *= 2

    lag_true = np.pi
    Y1 = f(X1) + np.random.randn(len(E1)) * E1
    Y2 = f(X2 + lag_true) + np.random.randn(len(E2)) * E2

    #::::::::::::::::::::
    # Plot Mock Signals

    plt.figure()
    plt.errorbar(X1, Y1, E1, fmt='none', capsize=2)
    plt.errorbar(X2, Y2, E2, fmt='none', capsize=2, c='tab:orange')
    plt.grid()
    plt.legend(["Main Signal", "Delayed Signal"])

    #::::::::::::::::::::
    #Do JITted ICCF fit
    Nlags = 512
    Nterp, Nboot = 1024, 1024
    print("Running Fit")
    lags = np.linspace(-X1.ptp(), X1.ptp(), Nlags)
    correls_jax = correlfunc_jax_vmapped(lags, X1, Y1, X2, Y2, Nterp)
    jax_samples = correl_func_boot_jax_wrapper_nomap(lags, X1, Y1, X2, Y2, E1, E2, Nterp=Nterp, Nboot=Nboot)

    #::::::::::::::::::::
    res_mean, res_std = jax_samples.mean(), jax_samples.std()
    print("Lag = %.2f +/- %.2f, consistent with true lag of %.2f at %.2f sigma" % (
    res_mean, res_std, lag_true, abs(lag_true - res_mean) / res_std))

    #::::::::::::::::::::
    # Plot Results

    plt.figure()
    plt.plot(lags, correls_jax, label = "Un-Bootstrapped ICCF Correlation")
    plt.axhline(0, c='k')
    plt.hist(jax_samples, density=True, bins=24, alpha=0.75, label = "ICCF Samples")

    plt.axvline(lag_true, c='k', ls='--', label = "True Lag")

    plt.axvline(res_mean, c='tab:red', ls='--', label = "ICCF Mode")
    plt.axvline(res_mean - res_std, c='tab:red', ls=':')
    plt.axvline(res_mean + res_std, c='tab:red', ls=':')


    plt.ylim(0, 1.5)
    plt.xlim(-lag_true, lag_true * 3)
    plt.xlabel("Lag")
    plt.ylabel("ICCF Correl (Normalized")
    plt.legend( loc= 'upper right')
    plt.grid()

    plt.show()