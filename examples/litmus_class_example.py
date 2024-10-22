import matplotlib.pyplot as plt

import litmus

mymock = litmus.mocks.mock(3)
mymock.plot()

my_model = litmus.models.GP_simple()

fitting_method = litmus.fitting_methods.SVI_scan(stat_model=my_model,
                                                  Nlags=32,
                                                  init_samples=5_000,
                                                  grid_bunching=0.8,
                                                  optimizer_args={'tol': 1E-3,
                                                                  'maxiter': 1024,
                                                                  'increase_factor': 1.1,
                                                                  },
                                                  reverse=False,
                                                  ELBO_Nsteps=256,
                                                  ELBO_Nsteps_init=200,
                                                  ELBO_particles=32,
                                                  ELBO_optimstep=0.005,
                                                  ELBO_threshold =1000.0,
                                                  debug=True
                                                  )

litmus_handler = litmus.LITMUS(fitting_method)

litmus_handler.add_lightcurve(mymock.lc_1)
litmus_handler.add_lightcurve(mymock.lc_2)

litmus_handler.fit()
litmus_handler.lag_plot()
litmus_handler.plot_parameters()
