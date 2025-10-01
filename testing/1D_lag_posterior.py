import matplotlib.pyplot as plt
from litmus_rm import *
import numpy as np

# ----------------
SUBPLOT = False
LIGHTCURVES = True
POSTERIOR = True
SAVEFIGS = False
# ----------------

'''
mock = litmus.mocks.mock_B.copy(
    E=[0.01, 0.1],
    cadence=[10, 30],
    lag=360,
    tau=200,
    season=180,
)
'''
mock = mocks.mock(lag=360,
                  E=[0.1, 0.25],
                  tau=200)
mock = mock(3)

lc_1, lc_2 = mock.lc_1, mock.lc_2
mock.plot()

model = litmus_rm.models.GP_simple()
data = model.lc_to_data(lc_1, lc_2)

fixed_params = mock.params()
del fixed_params['lag']

model.set_priors(fixed_params)

samps = model.prior_sample(10_000)
LL = model.log_density(samps, data)

X = samps['lag']
# Y = np.exp(LL-LL.max())
Y = LL

I = X.argsort()
X, Y = X[I], Y[I]

if POSTERIOR:
    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, c='midnightblue', lw=2, label = "Posterior")
    plt.plot(X, Y, c='w', lw=8, zorder=-1)

    if mock.season != 0:
        tmax = model.prior_ranges['lag'][1]
        nyears = int(tmax // (mock.season * 2) + 1)
        for i in range(nyears):
            plt.axvspan((i + 1 / 2 - 1 / 4) * mock.season * 2, (i + 1 - 1 / 4) * mock.season * 2,
                        ymin=0, ymax=1, alpha=0.25, color='royalblue',
                        zorder=-10,
                        label="Aliasing Seasons" if i == 0 else None)

    plt.xlabel("Lag, $\Delta t$")
    plt.ylabel("Log-Probability Density")

    plt.scatter(X[Y.argmax()], Y.max(), c='r', label="Maximum Posterior Density", zorder=100)

    if False:
        lags = [0, 180, mock.lag]
        idx = [np.argmin(np.abs(X - l)) for l in lags]
        plt.scatter(lags, Y[idx], c='b', marker='x')

    plt.axhline(Y.max() - np.log(100), c='k', ls='--', label="$1\%$ of max")
    plt.axvline(mock.lag, c='k', ls=':', label="True Lag, $%.0f \mathrm{d}$" %mock.lag)
    plt.xlim(*model.prior_ranges['lag'])
    Ywidth = (Y.max() - np.median(Y)) * 2
    plt.ylim(Y.max() - Ywidth * 2, Y.max() + Ywidth * 0.1)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.gca().set_xticklabels([0,200,400,600,800])
    if SAVEFIGS: plt.savefig("./1D_alising_log_posterior.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if LIGHTCURVES:
    lags = [0, 180, 360]
    titles = ["$\Delta t=0 \mathrm{ d}$, Bad Fit",
              "$\Delta t=180 \mathrm{ d}$, Ambiguous Fit",
              "$\Delta t=360 \mathrm{ d}$, Good Fit"
              ]
    f, a = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    for l, ax, t in zip(lags, a, titles):
        print(l)
        test_mock = mock.copy()
        test_mock.lag = l
        test_mock.corrected_plot(axis=ax,
                                 true_args={'c': ['navy', 'orchid'], 'label': [None, None]},
                                 series_args={'c': ['navy', 'orchid'], 'label': ["Continuum", "Response"]}
                                 )
        ax.grid()
        ax.set_title(t)
        if l == 180: ax.legend()
    f.supxlabel("Time (Days)")
    a[0].set_ylabel("Signal \n (Arb Units)")
    f.tight_layout()
    if SAVEFIGS: plt.savefig("./1D_alising_examples.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------
if SUBPLOT:
    f, [a1, a2] = plt.subplots(2, 1, figsize=(8, 4))

    mock.plot(axis=a1, show=False)
    a1.legend(["Continuum", "Response"])
    a1.grid()
    a1.set_xlim(0, mock.maxtime)
    a1.set_xlabel("Signal Time (Days)")
    a1.set_ylabel("Signal Strength \n (Arb Units)")

    a2.plot(X, Y)
    a2.set_xlabel("Lag, $\Delta t$")
    a2.set_ylabel("Log-Probability Density")
    a2.scatter(X[Y.argmax()], Y.max(), c='r', label="Maximum Posterior Density")
    a2.axhline(Y.max() - np.log(100), c='k', ls='--', label="$1\%$ of max")
    a2.set_xlim(*model.prior_ranges['lag'])
    a2.set_ylim(-600, Y.max() + Y.ptp() * 0.025)
    a2.legend()
    a2.grid()

    tmax = model.prior_ranges['lag'][1]
    nyears = int(tmax // 360 + 1)
    for i in range(nyears):
        a2.axvspan((i + 1 / 2 - 1 / 4) * 360, (i + 1 - 1 / 4) * 360,
                   ymin=0, ymax=1, alpha=0.25, color='tab:red',
                   zorder=-1,
                   label="Aliasing Seasons" if i == 0 else None)

    f.tight_layout()
    if SAVEFIGS: f.savefig(".Aliasing_Season_withcurve.pdf", dpi=300, bbox_inches='tight')
    f.show()
