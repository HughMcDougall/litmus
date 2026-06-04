"""
Contains handy methods for cleaning outliers from a lightcurve
"""
import matplotlib.pyplot as plt
import numpy as np
import numpyro

from litmus_rm.logging import logger
from litmus_rm.lightcurve import lightcurve
from litmus_rm._utils import isiter

from numpyro.infer import MCMC, NUTS
from jax.random import PRNGKey
import jax.numpy as jnp
import numpyro.distributions as dist
from scipy.stats import norm
from numpyro.contrib.nested_sampling import NestedSampler
from typing import Union


# --------------------------------------
class cleaner(logger):

    def __init__(self, lc: lightcurve, season: float = 180.0, sigma_cut: float = 3.0, N_plot: int = 128):
        super().__init__()
        self.lc: lightcurve = lc
        self.season: float = season
        self.sigma_cut: float = sigma_cut
        self.N_plot: int = N_plot

        self.lc_splits: [lightcurve] = []

        self.cut_results: dict = {}
        self.has_run: bool = False

    def season_split(self, season: float = None):
        """
        Indentifies the breaks in a lightcurve
        """
        if season is None: season = self.season
        if season == 0: return [self.lc]
        I = np.argwhere(np.diff(self.lc.T) > 0.5 * self.season).flatten()
        Tsplit = np.array([self.lc.T.min(), *self.lc.T[I], self.lc.T.max()])

        out = []
        for i in range(1, Tsplit.shape[0]):
            t1, t2 = Tsplit[i - 1], Tsplit[i]
            _lc = self.lc[np.argwhere((self.lc.T > t1) * (self.lc.T <= t2)).flatten()]
            out.append(_lc)

        return out

    def clean(self):
        # break lc's into seasons
        self.has_run = False
        lc_seasons = self.season_split(season=self.season)

        self.cut_results = []

        # Fit some regression & sigma for each

        for lc in lc_seasons:
            w = lc.E ** -2 if min(lc.E != 0) else np.ones_like(lc.Y)
            w /= w.sum()
            mu = np.average(lc.Y, weights=w)
            sig = np.average((lc.Y - mu) ** 2, weights=w) ** 0.5

            Icut = np.argwhere(abs(lc.Y - mu) > self.sigma_cut * sig).flatten()
            goodbad = np.array([i not in Icut for i in range(len(lc))])

            Tplot = np.linspace(lc.T.min(), lc.T.max(), self.N_plot)
            muplot = np.mean(lc.Y) * np.ones_like(Tplot)
            sigplot = np.std(lc.Y) * np.ones_like(Tplot)
            self.cut_results.append({"lc": lc,
                                     "goodbad": goodbad,
                                     "Tplot": Tplot,
                                     "mu": muplot,
                                     "sigma": sigplot}
                                    )
        self.has_run = True

    def get_cleaned(self):
        if not self.has_run: self.clean()
        Ts, Ys, Es = [], [], []
        for cr in self.cut_results:
            lc = cr["lc"]
            goodbad = cr["goodbad"]
            Ts.append(lc.T[goodbad == True])
            Ys.append(lc.Y[goodbad == True])
            Es.append(lc.E[goodbad == True])
        T, Y, E = np.concatenate(Ts), np.concatenate(Ys), np.concatenate(Es)
        return lightcurve(T, Y, E)

    def diagnostic_plot(self, show=True, title=None):
        if not self.has_run: self.clean()

        plt.figure(figsize=(12, 6))
        for j, cr in enumerate(self.cut_results):
            lc = cr["lc"]
            goodbad = cr["goodbad"]

            Tplot = cr["Tplot"]
            mu = cr["mu"]
            sig = cr["sigma"]

            plt.plot(Tplot, mu, ls='--', c='k')
            plt.fill_between(Tplot, mu - sig, mu + sig, alpha=0.25, color='grey',
                             label="Approx. Seasonal Spread ($1.0 \sigma$")
            plt.fill_between(Tplot, mu - sig * self.sigma_cut, mu + sig * self.sigma_cut, alpha=0.125, color='red',
                             label="$%.2f \sigma$ cut bounds (approx)" % self.sigma_cut)

            plt.errorbar(lc.T, lc.Y, yerr=lc.E, fmt='none', capsize=2, c='k')
            plt.scatter(lc.T[goodbad == True], lc.Y[goodbad == True], color='darkcyan', label="Retained Epochs")
            plt.scatter(lc.T[goodbad == False], lc.Y[goodbad == False], color='firebrick', label="Cut Epochs")

            if j == 0: plt.legend()

        if title is not None: plt.title(title)

        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.ylim(self.lc.Y.min() - self.lc.Y.std(), self.lc.Y.max() + self.lc.Y.std())
        if show: plt.show()
        return plt.gcf()


# --------------------------

class cleaner_polybayes(cleaner):
    def __init__(self, lc: lightcurve, season: float = 180.0, sigma_cut: float = 3.0, N_plot: int = 128,
                 npoly: Union[int, list[float]] = 3):
        super().__init__(lc=lc, season=season, sigma_cut=sigma_cut, N_plot=N_plot)
        self.npoly: Union[int, list[float]] = npoly

        # ----------------
        def numpyro_model(T, Y, E, npoly, coeffs_best, coeffs_uncert):
            Tgrid = jnp.tile(T, (npoly, 1))
            Tgrid = jnp.power(Tgrid, jnp.arange(npoly)[:, None])

            coeffs = numpyro.sample("coeffs", dist.Uniform(-coeffs_best * 10, coeffs_best * 10))
            mu_bg = numpyro.sample("mu_bg", dist.Normal(Y.mean(), Y.std() * 10))
            if True:
                weighting = numpyro.sample("weighting", dist.Exponential(jnp.log(10) * 10))
                weighting = jnp.min(jnp.array([1, weighting]))
            else:
                weighting = numpyro.sample("weighting", dist.Uniform(0, 0.25))

            sig_fg = numpyro.sample("sig_fg", dist.Uniform(0, Y.std() * 10 + E.max()))
            sig_bg = numpyro.sample("sig_bg", dist.Uniform(sig_fg if npoly == 1 else 0.0, Y.std() * 10 + E.max()))
            s1 = jnp.sqrt(jnp.square(E) + jnp.square(sig_fg))
            s2 = jnp.sqrt(jnp.square(E) + jnp.square(sig_bg))

            mu_fg = Tgrid.T @ coeffs

            w1 = -0.5 * jnp.square((Y - mu_fg) / s1) - jnp.log(s1) - 0.5 * jnp.log(2 * jnp.pi) + jnp.log(
                (1 - weighting))
            w2 = -0.5 * jnp.square((Y - mu_bg) / s2) - jnp.log(s2) - 0.5 * jnp.log(2 * jnp.pi) + jnp.log(weighting)

            LL = jnp.log(jnp.exp(w1) + jnp.exp(w2))
            numpyro.deterministic('affinity', jnp.square((Y - mu_fg) / s1))
            numpyro.factor('Y', LL.sum())

        try:
            Nlive = (4 + max(npoly) + 1) * 10 * 2
        except:
            Nlive = (4 + npoly + 1) * 10 * 2
        self.sampler = NestedSampler(numpyro_model,
                                     constructor_kwargs={'num_live_points': Nlive,
                                                         'max_samples': Nlive * 1000},
                                     termination_kwargs={'live_evidence_frac': 0.001}

                                     )

    def clean(self):
        self.has_run = False
        lc_seasons = self.season_split(season=self.season)

        if isiter(self.npoly):
            assert len(lc_seasons) == len(self.npoly), "Tried to specify %i poly orders for %i seasons" % (
                len(self.npoly), len(lc_seasons)
            )
            npoly_allseason = self.npoly
        else:
            npoly_allseason = np.ones(len(lc_seasons)).astype(int) * self.npoly

        self.cut_results = []

        for i, lc in enumerate(lc_seasons):
            self.msg_run("Cleaning lc %i of len %i" % (i, len(lc)))
            T, Y, E = [x.copy() for x in [lc.T, lc.Y, lc.E]]
            T -= T.mean()
            # Exit early if only one DP
            if len(lc) == 1:
                self.msg_run("lc only has one entry. Skipping.")
                Tplot = np.linspace(-self.season * 0.1, self.season * 0.1, self.N_plot) + lc.T[0]
                self.cut_results.append({"lc": lc,
                                         "goodbad": np.ones(len(lc)).astype(bool),
                                         "Tplot": Tplot,
                                         "mu": np.ones_like(Tplot) * lc.Y[0],
                                         "sigma": np.ones_like(Tplot) ** lc.E[0]}
                                        )

            # Determine order of regression
            if len(lc) <= 4:
                npoly_season = 1
            else:
                npoly_season = min(npoly_allseason[i], (len(lc) - 1) // 4)

            self.msg_run("Prepping fit with polynomial order %i" % npoly_season)

            # ----------------

            W = jnp.diag(E ** -2)
            Tgrid = np.tile(T, (npoly_season, 1))
            Tgrid = np.power(Tgrid, np.arange(npoly_season)[:, None])

            cov_coeff = jnp.linalg.inv(Tgrid @ (W @ Tgrid.T))
            coeffs_best = cov_coeff @ (Tgrid @ W @ Y.T)
            coeffs_uncert = jnp.diag(jnp.linalg.inv(Tgrid @ (jnp.diag((E ** 2 + Y.std() ** 2) ** -1) @ Tgrid.T)))

            self.sampler.run(PRNGKey(0), T, Y, E, npoly_season, coeffs_best, coeffs_uncert, )
            res = self.sampler.get_samples(PRNGKey(1), int(10_000))

            p_cut = norm.cdf(- self.sigma_cut)
            goodbad = norm.cdf(-res['affinity']).mean(axis=0) > p_cut

            Tplot = np.linspace(T.min(), T.max(), self.N_plot)
            Tplotgrid = np.tile(Tplot, (npoly_season, 1))
            Tplotgrid = np.power(Tplotgrid, np.arange(npoly_season)[:, None])
            pred = Tplotgrid.T @ res['coeffs'].T
            muplot = pred.mean(axis=1)
            sigplot = np.sqrt((pred - muplot[:, None]).var(axis=1) + np.mean(res['sig_fg']) ** 2)

            self.cut_results.append({"lc": lc,
                                     "goodbad": goodbad,
                                     "Tplot": Tplot + lc.T.mean(),
                                     "mu": muplot,
                                     "sigma": sigplot}
                                    )

        self.has_run = True


class cleaner_GRWbayes(cleaner):

    def clean(self):
        self.has_run = False

        from tinygp.kernels.quasisep import Exp
        from tinygp import GaussianProcess

        def model_function(self, T, Y, E) -> None:
            logtau = numpyro.sample('logtau', dist.Uniform(np.log(min(jnp.diff(T))), jnp.log(T.max() - T.min())))
            logamp = numpyro.sample('logtau', dist.Uniform(np.log(Y.std() / 100), np.log(Y.std() * 100)))
            jitter = numpyro.sample('jitter', dist.Uniform(0.0, 1.00))
            mu_fg = numpyro.sample("mu_bg", dist.Normal(Y.mean(), Y.std() * 10))

            mu_bg = numpyro.sample("mu_bg", dist.Normal(Y.mean(), Y.std() * 10))
            sig_bg = numpyro.sample("sig_bg", dist.Uniform(0, Y.std() * 10 + E.max()))

            weighting = numpyro.sample("weighting", dist.Uniform(0, 0.25))

            # Conversions to gp-friendly form
            amp, tau = jnp.exp(logamp), jnp.exp(logtau)

            diag = jnp.square(E) + jnp.square(jitter * amp)
            Y -= mu_fg

            # Build and sample GP
            kernel1 = Exp(scale=tau, sigma=amp * jnp.sqrt(1 - jnp.square(jitter)))
            kernel2 = Exp(scale=0.0, sigma=sig_bg)

            fg_dist = GaussianProcess(kernel1, T, diag=diag).numpyro_dist()
            bg_dist = GaussianProcess(kernel2, T, diag=diag).numpyro_dist()

            mixdist = numpyro.distributions.MixtureGeneral(
                numpyro.distributions.Categorical(jnp.array([weighting, 1 - weighting])),
                [
                    bg_dist,
                    fg_dist
                ]
            )

            w1 = fg_dist.log_prob(Y) + jnp.log((1 - weighting))
            w2 = bg_dist.log_prob(Y) + jnp.log(weighting)

            LL = jnp.log(jnp.exp(w1) + jnp.exp(w2))

            numpyro.deterministic('q', w1 - w2)
            numpyro.factor("Y", LL)

        self.has_run = True


if __name__ == "__main__":
    from litmus_rm.mocks import mock
    from datautils import get_lightcurves

    # mymock = mock()
    lc_1, lc_2, name, line = get_lightcurves(0, rootfol="../")

    mycleaner = cleaner_polybayes(lc_1, npoly=[1, *(np.ones(6).astype(int) * 4)], sigma_cut=3)

    lc_out = mycleaner.get_cleaned()
    mycleaner.diagnostic_plot()
