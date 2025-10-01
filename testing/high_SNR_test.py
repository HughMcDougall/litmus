
#------------------------------
import numpy as np
import matplotlib.pyplot as plt

import litmus_rm

mymock = litmus_rm.mocks.mock_C.copy(tau=np.exp(5.56), lag=106, cadence=(7, 7), E=[0.1, 0.1], season=0.0)

mymock = mymock(3)

mymock.plot()
plt.show()

model = litmus_rm.models.GP_simple()
model.debug = False
lc_1, lc_2 = mymock.lc_1, mymock.lc_2

if False:
    seed_params = model.find_seed(model.lc_to_data(lc_1, lc_2))[0]
else:
    seed_params = mymock.params()

print("\t Param \t Est \t True")
for key in seed_params.keys():
    print("\t %s: \t%.2f \t %.2f" % (key, seed_params[key], mymock.params()[key]))

fitting_method = litmus_rm.fitting_methods.hessian_scan(model,
                                                     Nlags=64,
                                                     init_samples=5_000,
                                                     grid_bunching=0.8,
                                                     optimizer_args_init={'tol': 1E-8,
                                                                          },

                                                     optimizer_args={'tol': 1E-5,
                                                                     'increase_factor':1.2
                                                                     },
                                                     reverse=False,
                                                     LL_threshold=20,
                                                     seed_params=seed_params,
                                                     precondition = 'half-eig'
                                                     )

litmus_handler = litmus_rm.LITMUS(fitting_method)
litmus_handler.add_lightcurve(lc_1)
litmus_handler.add_lightcurve(lc_2)

litmus_handler.prefit()

litmus_handler.fit()

litmus_handler.lag_plot()
litmus_handler.plot_parameters()
litmus_handler.diagnostic_plots()