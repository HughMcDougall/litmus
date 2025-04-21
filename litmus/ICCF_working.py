"""
ICCF_working.py

JAX-friendly working for performing the ICCF fit. To be called by a fitting method

todo - refactor / rename these functions
"""

# =================================================

import jax
import jax.numpy as jnp


# =================================================

def correl_jax(X1: [float], Y1: [float], X2: [float], Y2: [float], Nterp=1024) -> float:
    """
    Calculates a linearly interpolated correlation value for two data series {X1,Y1} and {X2,Y2}, with Nterp interpolation points

    :param X1: X values of series 1
    :param Y1: Y values of series 1
    :param X2: X values of series 2
    :param Y2: Y values of series 2
    :param Nterp: Number of samples to perform linear interpolation with
    :return: correlation coefficient if X1,X2 overlap and 0.0 if outside
    """

    # Find X values that are common to both arrays
    Xmin = jnp.max(jnp.array([jnp.min(X1), jnp.min(X2)]))
    Xmax = jnp.min(jnp.array([jnp.max(X1), jnp.max(X2)]))

    extrap = 0.0

    def f():
        X_interp = jnp.linspace(Xmin, Xmax, Nterp)
        Y1_interp = jnp.interp(X_interp, X1, fp=Y1, left=extrap, right=extrap)
        Y2_interp = jnp.interp(X_interp, X2, fp=Y2, left=extrap, right=extrap)
        out = jnp.corrcoef(x=Y1_interp, y=Y2_interp)[0][1]
        return (out)

    out = jax.lax.cond(Xmax > Xmin, f, lambda: 0.0)

    return (out)


correl_jax_jitted = jax.jit(correl_jax, static_argnames=["Nterp"])


#::::::::

def correlfunc_jax(lag: float, X1: [float], Y1: [float], X2: [float], Y2: [float], Nterp: int = 1024) -> [float]:
    """
    Like correl_jax, but with signal 2 delayed by some lag

    :param lag: Value to advance signal 2 by, i.e. X2 -> X2 + lag
    :param X1: X values of series 1
    :param Y1: Y values of series 1
    :param X2: X values of series 2
    :param Y2: Y values of series 2
    :param Nterp: Number of samples to perform linear interpolation with
    :return: correlation coefficient if X1, X2+lag overlap and 0.0 if outside
    """
    return (
        correl_jax(X1, Y1, X2 - lag, Y2, Nterp)
    )


correlfunc_jax_vmapped = jax.vmap(correlfunc_jax, in_axes=(0, None, None, None, None, None))
correlfunc_jax_vmapped.__doc__ += "Accepts array of lags for 'lag' param"


#::::::::

def correl_func_boot_jax(seed: int, lags: [float], X1: [float], Y1: [float], X2: [float], Y2: [float], E1: [float],
                         E2: [float], Nterp: int = 1024, N1: int = 2, N2: int = 2):
    """
    Finds the best fit lag for a single bootstrapped (subsampled & jittered) linterp correlation function

    :param seed: integer seed for randomisation
    :param lags: Array of lags to test at
    :param X1: X values of series 1
    :param Y1: Y values of series 1
    :param X2: X values of series 2
    :param Y2: Y values of series 2
    :param E1: E values of series 1
    :param E2: E values of series 2
    :param Nterp: Number of samples to perform linear interpolation with
    :param N1: Number of entries in the subsampled time series 1
    :param N2: Number of entries in the subsampled time series 2
    :return: correlation coefficient if X1, X2+lag overlap and 0.0 if outside
    """
    key = jax.random.key(seed)

    I1 = jax.random.choice(key=key, a=jnp.arange(X1.size), shape=(N1,), replace=False)
    I2 = jax.random.choice(key=key, a=jnp.arange(X2.size), shape=(N2,), replace=False)
    I1, I2 = jnp.sort(I1), jnp.sort(I2)

    X1p, X2p = X1[I1], X2[I2]
    Y1p = Y1[I1] + jax.random.normal(key, shape=(I1.size,)) * E1[I1]
    Y2p = Y2[I2] + jax.random.normal(key, shape=(I2.size,)) * E2[I2]
    correls = correlfunc_jax_vmapped(lags, X1p, Y1p, X2p, Y2p, Nterp)
    i_max = jnp.argmax(correls)
    return (lags[i_max])


correl_func_boot_jax_nomap = jax.jit(correl_func_boot_jax, static_argnames=["Nterp", "N1", "N2"])

correl_func_boot_jax = jax.vmap(correl_func_boot_jax,
                                in_axes=(0, None, None, None, None, None, None, None, None, None, None))
correl_func_boot_jax = jax.jit(correl_func_boot_jax, static_argnames=["Nterp", "N1", "N2"])


def correl_func_boot_jax_wrapper(lags: float, X1: [float], Y1: [float], X2: [float], Y2: [float], E1: [float],
                                 E2: [float], Nterp: int = 1024, Nboot: int = 512, r: float = jnp.exp(-1)):
    """
    Finds the best fit lag for a single bootstrapped (subsampled & jittered) linterp correlation function. Vmaps over all bootstrap realisations

    :param lags: Array of lags to test at
    :param X1: X values of series 1
    :param Y1: Y values of series 1
    :param X2: X values of series 2
    :param Y2: Y values of series 2
    :param E1: E values of series 1
    :param E2: E values of series 2
    :param Nterp: Number of samples to perform linear interpolation with
    :param Nboot: Number of bootstrap realisations to use
    :param r: fraction to subsample the number of time series values by
    :return: correlation coefficient if X1, X2+lag overlap and 0.0 if outside
    """
    seeds = jnp.arange(Nboot)
    N1, N2 = int(len(X1) * r), int(len(X2) * r)

    out = correl_func_boot_jax(seeds, lags, X1, Y1, X2, Y2, E1, E2, Nterp, N1, N2)

    return out


def correl_func_boot_jax_wrapper_nomap(lags, X1, Y1, X2, Y2, E1, E2, Nterp: int = 1024, Nboot: int = 512,
                                       r: float = jnp.exp(-1)) -> [float]:
    """
    Finds the best fit lag for a single bootstrapped (subsampled & jittered) linterp correlation function. Performs calcs in series

    :param lags: Array of lags to test at
    :param X1: X values of series 1
    :param Y1: Y values of series 1
    :param X2: X values of series 2
    :param Y2: Y values of series 2
    :param E1: E values of series 1
    :param E2: E values of series 2
    :param Nterp: Number of samples to perform linear interpolation with
    :param Nboot: Number of bootstrap realisations to use
    :param r: fraction to subsample the number of time series values by
    :return: correlation coefficient if X1, X2+lag overlap and 0.0 if outside
    """

    N1, N2 = int(len(X1) * r), int(len(X2) * r)

    out = [correl_func_boot_jax_nomap(seed, lags, X1, Y1, X2, Y2, E1, E2, Nterp, N1, N2) for seed in range(Nboot)]

    return jnp.array(out)
