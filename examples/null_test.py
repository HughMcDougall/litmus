'''
Demonstration of model comparison testing
'''

from litmus_rm import *

# ------
# MODELS
model_alt = litmus_rm.models.GP_simple()
model_alt_norm = litmus_rm.models.GP_simple_normalprior()
model_alt_norm.mu_lagpred = 4.3
model_null = litmus_rm.models.GP_simple_null()
model_whitenoise = litmus_rm.models.whitenoise_null()

MODELS = [model_alt, model_null, model_whitenoise]

# ------
# DATA
vague = True

if not vague:
    mock1 = litmus_rm.mocks.mock(1, lag=540, tau=200, E=[0.01, 0.1])
    mock2 = mock1.copy(seed=5)
    mock3 = litmus_rm.mocks.mock(8, lag=180, tau=0.001, E=[0.01, 0.1])
else:
    mock1 = litmus_rm.mocks.mock(3, lag=540, tau=50, E=[0.2, 0.5])
    mock2 = mock1.copy(seed=5)
    mock3 = litmus_rm.mocks.mock(8, lag=180, tau=0.001, E=[0.2, 0.5])

mock1_null, mock1_whitenoise = mock1.copy(), mock1.copy()
mock1_null.swap_response(mock2)
mock1_whitenoise.swap_response(mock3)

MOCKS = [mock1, mock1_null, mock1_whitenoise]
mock1.name = "Positive"
mock1_null.name = "Negative"
mock1_whitenoise.name = "Noise"
fig, axes = plt.subplots(len(MOCKS), 1, sharex=True, sharey=True)
for mock, ax in zip(MOCKS, axes):
    mock.plot(axis=ax, show=False,
              true_args={'c': ['navy', 'orchid'], 'label': [None, None]},
              series_args={'c': ['navy', 'orchid'], 'label': ["Continuum", "Response"]}
              )
    ax.grid()
fig.supxlabel("Time (Days)"), fig.supylabel("Signal (Arb Units)")
axes[0].set_title("True Lag Response"), axes[1].set_title("Decoupled Response"), axes[2].set_title(
    "White Noise Response")
axes[0].legend()
fig.tight_layout()
fig.savefig("./null_test_vague.pdf" if vague else "./null_test.pdf", dpi=300)
plt.show()

# ------
Z = []
FITTERS = []
i = 0
for mock in MOCKS:
    for model in MODELS:
        print(i)
        if i < len(Z):
            i += 1
            continue
        fitter = litmus_rm.fitting_methods.SVI_scan(model, Nlags=64, precondition="diag", ELBO_Nsteps=512,
                                                 grid_bunching=0.25)
        fitter.name = mock.name + " " + model.name
        fitter.verbose = True
        fitter.debug = True
        fitter.fit(mock.lc_1, mock.lc_2)

        FITTERS.append(fitter)
        try:
            Z.append(fitter.get_evidence())
        except:
            Z.append(None)
        i += 1

print("ALL FITTING DONE. PRESENTING EVIDENCE RATIOS")

LZ = np.log10(np.array(Z)[:, 0]).reshape(len(MOCKS), len(MODELS))
LZ = np.round(LZ, 2)

print("Log10 Z")
for i, row in enumerate(LZ):
    print(MOCKS[i].name, '\t', row)

print("\n\n log Rat vs null")
for i, row in enumerate(LZ - np.tile(LZ[:, 2], [len(MODELS), 1]).T):
    print(MOCKS[i].name, '\t', row)

for fitter in FITTERS:
    samplags = fitter.get_samples(50_000)['lag']
    mu, sig = samplags.mean(), samplags.std()
    mumed = np.median(samplags)
    print("For fitter %s, recovered lag is " % fitter.name, end=' ')
    if abs(mumed - mu) < 2 * sig:
        print("%.2f +/- %.2f" % (mu, sig))
    else:
        print("Unconverged")

for fitter in FITTERS:
    print("Fit done. Making litmus")
    lt = LITMUS(fitter)

    # print("Doing param plot")
    lt.plot_parameters(prior_extents=True)
    print("Doing lag plot")
    lt.lag_plot()
print("Script finished")
