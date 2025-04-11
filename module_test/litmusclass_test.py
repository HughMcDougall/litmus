if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import matplotlib
    from litmus.litmusclass import *

    matplotlib.use('module://backend_interagg')

    #::::::::::::::::::::
    # Mock Data
    from litmus.mocks import *

    # mymock = mymock.copy(E=[0.05,0.1], lag=300)
    mymock = mock(cadence=[7, 30], E=[0.15, 1.0], season=180, lag=180 * 3, tau=200.0)
    # mymock=mock_B
    mymock.plot()
    mymock(10)
    plt.show()

    lag_true = mymock.lag

    test_model = models.GP_simple()
    test_model.set_priors(mock.params())

    seed_params = {}

    #::::::::::::::::::::
    # Make Litmus Object
    fitting_method = fitting_methods.hessian_scan(stat_model=test_model,
                                                  Nlags=24,
                                                  init_samples=5_000,
                                                  grid_bunching=0.8,
                                                  optimizer_args={'tol': 1E-3,
                                                                  'maxiter': 512,
                                                                  'increase_factor': 1.2,
                                                                  },
                                                  reverse=False,
                                                  ELBO_Nsteps=300,
                                                  ELBO_Nsteps_init=200,
                                                  ELBO_particles=24,
                                                  ELBO_optimstep=0.014,
                                                  seed_params=seed_params,
                                                  debug=True
                                                  )

    test_litmus = LITMUS(fitting_method)

    test_litmus.add_lightcurve(mymock.lc_1)
    test_litmus.add_lightcurve(mymock.lc_2)

    print("\t Fitting Start")
    test_litmus.fit()
    print("\t Fitting complete")

    test_litmus.plot_parameters()
    test_litmus.lag_plot()
    test_litmus.diagnostic_plots()
