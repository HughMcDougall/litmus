'''
Demonstration of model comparison testing
'''

from litmus import *

# ------
# MODELS
model_alt = litmus.models.GP_simple()
model_null_delta = litmus.models.GP_simple_null()
# model_null_delta.set_priors({'lag': [0.0, 0.0]})

MODELS = [model_null_delta]

# ------
# DATA
mock1 = litmus.mocks.mock(1)
mock2 = litmus.mocks.mock(20)

mock1_null, mock2_null = mock1.copy(), mock2.copy()
mock1_null.swap_response(mock2_null)

MOCKS = [mock1, mock1_null]
mock1.name = "Positive"
mock1_null.name = "Negative"
# MOCKS = [mock1_null]

# ------
Z = []
FITTERS = []
for mock in MOCKS:
    mock.plot()

    for model in MODELS:
        fitter = litmus.fitting_methods.hessian_scan(model, Nlags = 32, precondition="diag", ELBO_Nsteps = 256)
        fitter.verbose=True
        fitter.debug=True
        fitter.fit(mock.lc_1, mock.lc_2)

        FITTERS.append(fitter)
        try:
            Z.append(fitter.get_evidence())
        except:
            Z.append(None)

for fitter in FITTERS:
    print("Fit done. Making litmus")
    lt = LITMUS(fitter)
    print("Doing param plot")
    lt.plot_parameters(prior_extents=True)
    print("Doing lag plot")
    lt.lag_plot()

print("ALL FITTING DONE. PRESENTING EVIDENCE RATIOS")

LZ = np.log10(np.array(Z)[:, 0]).reshape(len(MODELS), len(MOCKS))
LZ = np.round(LZ,2)
LZ = LZ.T

print("Log10 Z")
for i, row in enumerate(LZ):
    print(MOCKS[i].name,'\t', row)

print("\n\n log Rat vs null")
for i, row in enumerate(LZ - np.tile(LZ[:, -1], [len(MOCKS), 1]).T):
    print(MOCKS[i].name,'\t', row)