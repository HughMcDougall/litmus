import matplotlib.pyplot as plt

import litmus

mymock = litmus.mocks.mock(3, cadence = [30,30], E = [0.5,0.5])
mymock.plot()

my_model = litmus.models.GP_simple()

fitting_method = litmus.fitting_methods.hessian_scan(stat_model=my_model,
                                                  Nlags=64,
                                                  init_samples=5_000,
                                                  grid_bunching=0.5,
                                                  grid_firstdepth = 10,
                                                  optimizer_args={'tol': 1E-3,
                                                                  'maxiter': 2048,
                                                                  'increase_factor': 1.2,
                                                                  },
                                                  LL_threshold = 100,
                                                  reverse=False,

                                                  debug=True
                                                  )

litmus_handler = litmus.LITMUS(fitting_method)

litmus_handler.add_lightcurve(mymock.lc_1)
litmus_handler.add_lightcurve(mymock.lc_2)

litmus_handler.fit()
litmus_handler.lag_plot()
litmus_handler.plot_parameters()
