"""
litmus_rm.nestedsampling.py
contains wrappers for different

"""
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey
from litmus_rm.logging import logger
from matplotlib import figure
import matplotlib.pyplot as plt
from matplotlib import get_backend, use
from jaxns.internals.log_semiring import cumulative_logsumexp

# -------------------------------------
try:
    import jaxns
    import tensorflow_probability.substrates.jax.distributions as tfpd

    has_jaxns = True
except ImportError:
    has_jaxns = False
    pass

try:
    import dill
    import dynesty
    from dynesty import plotting as dyplot
    from multiprocessing import Pool

    # from dynesty.pool import Pool

    dynesty.utils.pickle_module = dill

    has_dynesty = True
except ImportError:
    has_dynesty = False
    pass

try:
    import pypolychord

    has_polychord = True
except ImportError:
    has_polychord = False
    pass

_samplertypes = ["JAXNS", "dynesty_static", "dynesty_dynamic", "polychord"]


# -------------------------------------
class NS_wrapper(logger):
    """
    Generic wrapper on different nested sampler methods so that call types are consistent
    """

    def __init__(self, loglike_func, prior_func, ndim: int, addtl_args ={}, Nlive: int = None, tol: float = 1E-2, max_samples=None,
                 max_evals=None, num_parallel: int = 1, verbose=True, warn=True, debug=0, seed=1):

        super().__init__(verbose=verbose, debug=debug, warn=warn)
        if Nlive is None: Nlive = 50 * ndim
        if max_samples is None: max_samples = Nlive * 100
        if max_evals is None: max_evals = max_samples * 1000

        assert num_parallel > 0, "num_parallel must be > 0"
        assert ndim > 0, "ndim must be > 0"

        self.loglike_func = loglike_func
        self.prior_func = prior_func
        self.ndim: int = ndim
        self.Nlive: int = Nlive
        self.tol: float = tol
        self.seed: int = seed
        self.Ndead: int = 0

        self.max_samples: int = max_samples
        self.max_evals: int = max_evals
        self.num_parallel: int = num_parallel
        self.addtl_args = addtl_args

        self.has_run = False

        self.sampler = None
        self.results = None
        self.termination = None

    def run(self):
        """Runs the internal fitter"""
        self.has_run = True

    @property
    def live_samples(self) -> np.ndarray:
        """Returns the vectorized dead samples"""
        return np.zeros([self.Nlive, self.ndim])

    @property
    def log_weights(self) -> np.ndarray:
        """Fractional evidence contributions from each live sample"""
        return np.zeros([self.Nlive])

    @property
    def log_live_densities(self) -> np.ndarray:
        """Returns the unit-cube posterior densities of the dead points"""
        return np.zeros([self.Nlive])

    @property
    def log_live_volumes(self) -> np.ndarray:
        """Returns the unit-cube differential volumes associated with each dead sample"""
        return np.zeros([self.Nlive])

    @property
    def log_Z_and_err(self) -> np.ndarray:
        """Returns the log evidence and uncertainty"""
        return np.array([0.0, 0.0])

    @property
    def ESS(self) -> float:
        """
        Returns the effective sample size ESS = sum(W)^2 / sum(W^2)
        """
        return 0.0

    def posterior_samples(self, N, seed=None):
        if seed is not None: np.random.seed(seed)
        logw = self.log_live_densities + self.log_live_densities
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        I = np.random.choice(np.arange(len(w)), size=N, replace=True, p=w)
        out = self.live_samples[I, :]
        return out

    # -------------------------------------

    def diagnostic_plot(self) -> figure:
        """
        Returns a matplotlib figure with diagnostic plots
        """
        f = plt.figure()
        return f


# ---------------------------------------
class NS_JAXNS(NS_wrapper):

    def __init__(self, loglike_func, prior_func, ndim, *args, **kwargs):
        super().__init__(loglike_func, prior_func, ndim, *args, **kwargs)

        _jaxnsmodel = jaxns.Model(prior_model=self.prior_func,
                                  log_likelihood=self.loglike_func,
                                  )

        self._jaxnstermination = jaxns.TerminationCondition(
            dlogZ=self.tol,
            max_samples=self.max_samples,
        )

        self.sampler = jaxns.NestedSampler(
            model=_jaxnsmodel,
            max_samples=self.max_samples,
            verbose=self.debug,
            num_live_points=self.Nlive,
            num_parallel_workers=self.num_parallel,
            difficult_model=True,
            **self.addtl_args
        )

        self.has_run = False

    def run(self):
        self.msg_run("Running JAXNS sampler...", lvl=2)
        termination_reason, state = self.sampler(PRNGKey(self.seed),
                                                 self._jaxnstermination)

        self.msg_run("Running JAXNS done...", lvl=2)

        self.results = self.sampler.to_results(
            termination_reason=termination_reason,
            state=state
        )

        self.Ndead = len(self.results.log_posterior_density)

        self.has_run = True

    @property
    def live_samples(self) -> np.ndarray:
        if self.has_run:
            return self.results.samples['x']
        else:
            return np.zeros([self.Nlive, self.ndim])

    @property
    def log_weights(self) -> np.ndarray:
        if self.has_run:
            return self.results.log_dp_mean
        else:
            return np.zeros([self.Nlive])

    @property
    def log_live_densities(self) -> np.ndarray:
        if self.has_run:
            return self.results.log_posterior_density
        else:
            return np.zeros([self.Nlive])

    @property
    def log_live_volumes(self) -> np.ndarray:
        if self.has_run:
            return self.results.log_dp_mean + self.log_Z_and_err[0] - self.log_live_densities
        else:
            return np.zeros([self.Nlive])

    @property
    def log_Z_and_err(self) -> np.ndarray:
        if self.has_run:
            return np.array([self.results.log_Z_mean, self.results.log_Z_uncert])
        else:
            return np.zeros([2])

    @property
    def ESS(self) -> float:
        if self.has_run:
            logw = self.results.log_dp_mean
            w = np.exp(logw - logw.max())
            out = w.sum() ** 2 / (w ** 2).sum()
            return out
        else:
            return 0.0

    def posterior_samples(self, N, seed=None):
        return jaxns.resample(
            key=PRNGKey(self.seed if seed is None else seed),
            samples=self.live_samples,
            log_weights=self.log_weights,
            S=N
        )

    def diagnostic_plot(self) -> figure:
        num_samples = int(self.results.total_num_samples)
        f, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
        log_X = np.asarray(self.results.log_X_mean[:num_samples])
        num_live_points_per_sample = np.asarray(self.results.num_live_points_per_sample[:num_samples])
        log_L = np.asarray(self.results.log_L_samples[:num_samples])
        max_log_likelihood = np.max(log_L)
        log_dp_mean = np.asarray(self.results.log_dp_mean[:num_samples])
        log_cum_evidence = cumulative_logsumexp(log_dp_mean)
        cum_evidence = np.exp(log_cum_evidence)
        log_Z_mean = np.asarray(self.results.log_Z_mean)
        num_likelihood_evaluations_per_sample = np.asarray(
            self.results.num_likelihood_evaluations_per_sample[:num_samples])
        mean_efficiency = np.exp(self.results.log_efficiency)

        efficiency = 1. / num_likelihood_evaluations_per_sample

        # Plot the number of live points
        axs[0].plot(-log_X, num_live_points_per_sample, c='black')
        axs[0].set_ylabel(r'$n_{\rm live}$')
        # detect if too small log likelihood
        rel_log_L = log_L - max_log_likelihood
        axs[1].plot(-log_X, np.exp(rel_log_L), c='black')
        axs[1].axhline(1., color='black', ls='dashed',
                       label=rf"$\log L_{{\rm max}}={max_log_likelihood:.1f}$")
        axs[1].set_ylabel(r'$L/L_{\rm max}$')
        axs[1].legend()
        axs[2].plot(-log_X, np.exp(log_dp_mean), c='black')
        axs[2].axvline(-self.results.H_mean, color='black', ls='dashed',
                       label=rf'$-H={-self.results.H_mean:.1f}$')
        axs[2].set_ylabel(r'$Z^{-1}L dX$')
        axs[2].legend()
        axs[3].plot(-log_X, cum_evidence, c='black')
        axs[3].axhline(1., color='black', ls='dashed',
                       label=rf"$\log Z={log_Z_mean:.1f}$")
        axs[3].set_ylabel(r'$Z(\lambda > L)/Z$')
        axs[3].legend()
        axs[4].scatter(-log_X, efficiency, s=2, c='black')

        axs[4].axhline(mean_efficiency, color='black', ls='dashed',
                       label=f'avg. eff.={mean_efficiency:.3f}')
        axs[4].set_ylabel("sampler efficiency")
        axs[4].set_ylim(0., 1.05)
        axs[4].legend()
        axs[4].set_xlabel(r'$- \log X$')
        return f


# ---------------------------------------
class NS_dynestystatic(NS_wrapper):

    def __init__(self, loglike_func, prior_func, ndim, *args, **kwargs):
        super().__init__(loglike_func, prior_func, ndim, *args, **kwargs)

        if self.num_parallel > 1:
                p = Pool(self.num_parallel)

                self.sampler = dynesty.NestedSampler(loglikelihood=loglike_func, prior_transform=prior_func,
                                                     ndim=self.ndim, nlive=self.Nlive,
                                                     pool=p,
                                                     queue_size=self.num_parallel, **self.addtl_args
                                                     )
        else:
            self.sampler = dynesty.NestedSampler(self.loglike_func, self.prior_func, ndim=self.ndim, nlive=self.Nlive,
                                                 pool=None,
                                                 queue_size=1, **self.addtl_args
                                                 )

        self.has_run = False

    def run(self):
        self.msg_run("Running DyNesty Static sampler...", lvl=2)

        self.sampler.run_nested(maxiter=self.max_evals, dlogz=self.tol)
        self.results = self.sampler.results

        self.Ndead = len(self.results.logwt)

        self.has_run = True

    @property
    def live_samples(self) -> np.ndarray:
        if self.has_run:
            return self.results.samples
        else:
            return np.zeros([self.Nlive, self.ndim])

    @property
    def log_live_densities(self) -> np.ndarray:
        if self.has_run:
            return self.results.logwt - self.results.logvol
        else:
            return np.zeros([self.Nlive])

    @property
    def log_live_volumes(self) -> np.ndarray:
        if self.has_run:
            return self.results.logvol
        else:
            return np.zeros([self.Nlive])

    @property
    def log_Z_and_err(self) -> np.ndarray:
        if self.has_run:
            return np.array([self.results.logz[-1], self.results.logzerr[-1]])
        else:
            return np.zeros([2])

    @property
    def ESS(self) -> float:
        if self.has_run:
            logw = self.results.logwt
            w = np.exp(logw - logw.max())
            out = w.sum() ** 2 / (w ** 2).sum()
            return out
        else:
            return 0.0

    def posterior_samples(self, N, seed=None):
        try:
            if seed is not None: np.random.seed(seed)
        except ValueError:
            print("!!!!!!!!!!!!!!", seed, "\n\n\n")
        pos_samps = self.results.samples_equal()
        return pos_samps[np.random.choice(np.arange(pos_samps.shape[0]), N, replace=True), :]

    def diagnostic_plot(self) -> figure:
        return dyplot.runplot(self.results)[0]


class NS_dynestydynamic(NS_dynestystatic):

    def __init__(self, loglike_func, prior_func, ndim, N_effective=1_000, *args, **kwargs):
        super().__init__(loglike_func, prior_func, ndim, *args, **kwargs)

        self.sampler = dynesty.DynamicNestedSampler(self.loglike_func, self.prior_func, ndim=self.ndim,
                                                    pool=Pool(self.num_parallel, loglike_func,
                                                              prior_func) if self.num_parallel > 1 else None,
                                                    queue_size=self.num_parallel if self.num_parallel > 1 else None,
                                                    **self.addtl_args)
        self.N_effective: int = N_effective

        self.has_run = False

    def run(self):
        self.msg_run("Running DyNesty Dynamic sampler...", lvl=2)
        self.sampler.run_nested(nlive_init=self.Nlive, n_effective=self.N_effective, dlogz_init=self.tol)
        self.results = self.sampler.results

        self.Ndead = len(self.results.logwt)

        self.has_run = True


if __name__ == "__main__":

    print("Testing nestedsampling module")
    import jax.numpy as jnp

    # Define the dimensionality of our problem.
    ndim = 3

    # Define our 3-D correlated multivariate normal likelihood.
    C = np.identity(ndim)  # set covariance to identity matrix
    C[C == 0] = 0.95  # set off-diagonal terms
    Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
    lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
                    np.log(np.linalg.det(C)))  # ln(normalization)

    lo = np.ones(ndim) * -10.0
    hi = np.ones(ndim) * 10.0

    lo[-1], hi[-1] = 0.0, 0.0


    def loglike_jaxns(x):
        """The log-likelihood function."""

        return -0.5 * jnp.dot(x, jnp.dot(Cinv, x)) + lnorm


    def ptform_jaxns():
        x = yield jaxns.Prior(tfpd.Uniform(low=lo, high=hi), name='x')
        return x


    def loglike_dynesty(x):
        """The log-likelihood function."""

        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm


    # Define our uniform prior.
    def ptform_dynesty(u):
        """Transforms samples `u` drawn from the unit cube to samples to those
        from our uniform prior within [-10., 10.) for each variable."""

        return (hi - lo) * u + lo


    # --------------------
    fitter_jaxns = NS_JAXNS(loglike_jaxns, ptform_jaxns, ndim=3, Nlive=2_000)
    fitter_dyn1 = NS_dynestystatic(loglike_dynesty, ptform_dynesty, ndim=3, Nlive=2_000, num_parallel=2)

    fig, axes = plt.subplots(2, 1)
    for fit, lab in zip([fitter_jaxns, fitter_dyn1], ['JAXNS', 'DyNesty']):
        print("Results for %s" % lab)
        fit.run()

        print("logZ = %.2f +/- %.2f" % (fit.log_Z_and_err[0], fit.log_Z_and_err[1]))
        print("ESS = %.1f" % fit.ESS)
        print("total samps = %i / %i max samples" % (fit.Ndead, fit.max_samples))
        X, Y = fit.live_samples[:, :2].T
        Xp, Yp = fit.posterior_samples(1_000)[:, :2].T

        f = fit.diagnostic_plot()
        f.show()

        axes[0].scatter(X, Y, s=5, label=lab)
        axes[1].scatter(Xp, Yp, s=5, label=lab, alpha=0.3)

    for ax in axes: ax.grid()
    fig.show()
