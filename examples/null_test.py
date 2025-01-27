'''
Demonstration of model comparison testing
'''

from litmus import *

# ------
# MODELS
model_alt = litmus.models.GP_simple()
model_alt_norm = litmus.models.GP_simple_normalprior()
model_alt_norm.mu_lagpred = 4.3
model_null = litmus.models.GP_simple_null()
model_whitenoise = litmus.models.whitenoise_null()

MODELS = [model_alt, model_alt_norm, model_null, model_whitenoise]

# ------
# DATA
mock1 = litmus.mocks.mock(1, lag = 540, tau=200, E = [0.1,0.25])
mock2 = mock1.copy(seed=2)
mock3 = litmus.mocks.mock(2, lag = 180, tau=0.001, E = [0.1,0.25])

mock1_null, mock1_whitenoise = mock1.copy(), mock1.copy()
mock1_null.swap_response(mock2)
mock1_whitenoise.swap_response(mock3)

MOCKS = [mock1, mock1_null, mock1_whitenoise]
mock1.name = "Positive"
mock1_null.name = "Negative"
mock1_whitenoise.name = "Noise"
fig, axes = plt.subplots(len(MOCKS), 1, sharex=True, sharey=True)
for mock, ax in zip(MOCKS, axes):
    mock.plot(axis=ax, show=False)
    ax.grid()
plt.show()

# ------
Z = []
FITTERS = []
i=0
for mock in MOCKS:
    for model in MODELS:
        print(i)
        if i<len(Z):
            i += 1
            continue
        fitter = litmus.fitting_methods.hessian_scan(model, Nlags=64, precondition="diag", ELBO_Nsteps=512, grid_bunching = 0.25)
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
    if abs(mumed - mu )< 2 * sig:
        print("%.2f +/- %.2f" %(mu,sig))
    else:
        print("Unconverged")


for fitter in FITTERS:
    print("Fit done. Making litmus")
    lt = LITMUS(fitter)

    # print("Doing param plot")
    # lt.plot_parameters(prior_extents=True)
    print("Doing lag plot")
    lt.lag_plot()
print("Script finished")