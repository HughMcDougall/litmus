from litmus import *
from litmus.fitting_methods import fitting_procedure
from copy import copy as shallowcopy

mock = mocks.mock(1, cadence=[60, 60], E=[0.25, 0.25], season=0)
mock.plot()
lc_1, lc_2 = mock.lc_1, mock.lc_2

# ----
model = models.GP_simple()
model.set_priors(mock.params())
model.set_priors({
    'lag': [0, 500],
    'logtau': [0, 10]
})
data = model.lc_to_data(lc_1, lc_2)

# ----
Nlags = 64
grid_bunching = 0.5
HS = fitting_methods.hessian_scan(model, Nlags=Nlags, grid_bunching=grid_bunching,
                                  optimizer_args={'tol': 1E-2,
                                                  'maxiter': 32,
                                                  'increase_factor': 1.5}
                                  )
HS.prefit(lc_1, lc_2)

SS1 = fitting_methods.SVI_scan(model, Nlags=Nlags, grid_bunching=grid_bunching, seed_params=HS.estmap_params,
                               ELBO_Nsteps=512 * 3)
SS2 = fitting_methods.SVI_scan(model, Nlags=Nlags, grid_bunching=grid_bunching, seed_params=HS.estmap_params,
                               ELBO_Nsteps=512 * 3,
                               ELBO_particles=SS1.ELBO_particles * 2)
SS3 = fitting_methods.SVI_scan(model, Nlags=Nlags, grid_bunching=grid_bunching, seed_params=HS.estmap_params,
                               ELBO_Nsteps=512 * 3,
                               ELBO_particles=SS1.ELBO_particles * 4)
PS = fitting_methods.prior_sampling(model, Nsamples=50_000)

print("Doing Hessian Scan")
HS.fit(lc_1, lc_2)
# HS.refit()
print("Doing SVI Scan 1")
SS1.fit(lc_1, lc_2)
print("Doing SVI Scan 2")
#SS2.fit(lc_1, lc_2)
print("Doing SVI Scan 3")
#SS3.fit(lc_1, lc_2)
print("Doing Prior Sampling")
#PS.fit(lc_1, lc_2)
print("All fits done.")

labels = ["HS", "SS1"]
for S, label in zip([HS, SS1, SS2, SS3, PS], labels):
    print(label)
    try:
        print(S.get_evidence())
    except:
        print("Unconverged")
        continue

for S, label in zip([HS, SS1, SS2, SS3], labels):
    plt.scatter(S.scan_peaks['lag'], S.log_evidences, label=label)
    plt.axhline(S.log_evidences.max() -2, ls='--')
plt.grid()
plt.legend()
plt.show()

print("\n\n Run Complete.")
