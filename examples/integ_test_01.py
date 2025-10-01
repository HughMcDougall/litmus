'''
An attempt to ensure normalization is correct and that the constrained / unconstrained
jacobian factor is being applied properly
'''
import litmus_rm.fitting_methods
from litmus_rm import *

model_1 = litmus_rm.models.dummy_statmodel()

model_2 = litmus_rm.models.dummy_statmodel()
model_2.set_priors({'lag': 250.0})
model_3 = litmus_rm.models.dummy_statmodel()
model_3.set_priors({'lag': [250.0 - 0.0001, 250.0 + 0.0001]})

model_4 = litmus_rm.models.dummy_statmodel()
model_4.set_priors({'logtau': 0.5})
model_5 = litmus_rm.models.dummy_statmodel()
model_5.set_priors({'logtau': [0.5 - 0.00001, 0.5 + 0.00001]})

model_6 = litmus_rm.models.dummy_statmodel()
model_6.set_priors({'logtau': [-100.0, 100.0],
                    'logamp': [-30.0, 30.0]})

model_7a = litmus_rm.models.dummy_statmodel()
model_7a.set_priors({'lag': [-1000, 1000.0],
                     'logtau': [-10.0, 10.0]})
model_7b = litmus_rm.models.dummy_statmodel()
model_7b.set_priors({'lag': [-1000, 1000.0],
                     'logtau': [-20.0, 20.0]})

model_7c = litmus_rm.models.dummy_statmodel()
model_7c.set_priors({'lag': [-1000, 1000.0],
                     'logtau': [-30.0, 30.0]})

#model_names = ["normal", "fixed_lag", "narrow_lag", "fix_logtau", "narrow_logtau", "wide_prior", "wide_lag"]
#models = [model_1, model_2, model_3, model_4, model_5, model_6]
models = [model_1, model_7a, model_7b, model_7c ]
model_names = ["normal"] + list('abc')

mock = litmus_rm.mocks.mock_C()
lc_1, lc_2 = mock.lc_1, mock.lc_2

Z = []
LT = []
FM = []
# --------------------------------
for model, name in zip(models, model_names):
    data = model.lc_to_data(lc_1, lc_2)

    PS = litmus_rm.fitting_methods.prior_sampling(model, Nsamples=32 ** len(model.free_params()))

    HS1 = litmus_rm.fitting_methods.hessian_scan(model, Nlags=64)
    HS2 = litmus_rm.fitting_methods.hessian_scan(model, Nlags=64, interp_scale = 'linear')
    HS2.name += "-Linear Interp Scale"

    SV = litmus_rm.fitting_methods.SVI_scan(model, Nlags=64, ELBO_Nsteps = 256, ELBO_particles = 128, constrained_domain = False)

    NS = litmus_rm.fitting_methods.nested_sampling(model, num_live_points =800, max_samples = 200_000)
    methods = [HS1, HS2, NS]

    for method in methods:
        print("-" * 23)
        method.verbose = True
        method.debug = True
        method.name += "-" + name
        print(method.name)
        while isinstance(method, litmus_rm.fitting_methods.hessian_scan):
            method.prefit(lc_1, lc_2)
            if not np.isnan(model.log_density(method.estmap_params, data)): break
        method.fit(lc_1, lc_2)
        FM.append(method)
        print("-" * 23)
    print("Model Done.")
print("All Done")

print("Doing Evidences")
Z = []
for method in FM:
    print("Evidence:\t", method.get_evidence())
    Z.append(method.get_evidence())

LZ = np.log10(np.array(Z)[:, 0]).reshape(len(models), len(methods))
LZ = np.round(LZ,2)

print("Log10 Z")
for i, row in enumerate(LZ):
    print(model_names[i],'\t', row)

print("\n\n log Rat vs NS")
for i, row in enumerate(LZ - np.tile(LZ[:, -1], [len(methods), 1]).T):
    print(model_names[i],'\t', row)

print("\n\n log Rat vs Model 1")
for i, row in enumerate(LZ - np.tile(LZ[0, :], [len(models), 1])):
    print(model_names[i],'\t', row)

'''
for i in range(len(models)):
    f, ([a1,a2,a3], [b1,b2,b3]) = plt.subplots(2,3, sharex=True)
    for fm in [FM[3*i+0],FM[3*i+1]]:
        a1.plot(fm.scan_peaks['lag'], fm.log_evidences, '.--')
        a2.plot(fm.scan_peaks['lag'], fm.diagnostic_ints, '.--')
        a3.plot(fm.scan_peaks['lag'], fm.diagnostic_tgrads, '.--')

    a1.set_title("Log Evidence")
    a2.set_title("Integrals")
    a3.set_title("tgrad")
    select = np.where(FM[1].converged * FM[0].converged)[0]
    X = FM[0].lags[select]
    select = select[X.argsort()]
    X.sort()
    b1.plot(X, FM[1].log_evidences[select] - FM[0].log_evidences[select], '.--')
    b2.plot(X, FM[1].diagnostic_ints[select] - FM[0].diagnostic_ints[select], '.--')
    b3.plot(X, FM[1].diagnostic_tgrads[select] - FM[0].diagnostic_tgrads[select], '.--')
    for b in [b1,b2,b3]:
        b.axhline(y=0, linestyle='--', c='k', zorder=-1)
        b.set_ylim(-10,10)
        pass
    f.tight_layout()
    plt.show()
'''

'''
print("Doing Plots")
for method in FM:
    print(method.name)
    lt = LITMUS(method)
    LT.append(lt)
    lt.plot_parameters(prior_extents=True)
'''