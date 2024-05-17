'''
litmus.py

Contains the main litmus object class, which acts as a user-friendly interface with the models statistical models
and fitting procedure.

Currently only a placeholder. To be filled out later

vA - 13/5

'''
# ============================================
# IMPORTS
import sys

import models
import fitting_methods as fitprocs
import numpy as np
import jax.numpy as jnp
from lightcurve import lightcurve

#=========================================================
# LITMUS (Fit Handler)
#=========================================================

class litmus(object):
    '''
    Wrapper for lag recovery methods and configuration
    TODO
    - parameter search ranges
    - add lightcurve
    - 'fit_lag()'
    - 'plot_lightcurves'
    '''

    def __init__(self, model = None, fitproc = None):
        self.model = model
        self.fitproc = fitproc

        self.lightcurves = {}

        self.main_stream = sys.stdout
        self.err_stream  = sys.stderr

        return

    def add_lightcurve(self, data):
        self.lightcurves.append(data)
        return

    def remove_lightcurve(self):
        '''todo - fill this out'''
        return

    def fit_lag(self):
        self.fitproc
        return


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    #::::::::::::::::::::
    # Mock Data
    print("Making Mock Data")
    f = lambda x: np.exp(-((x - 8) / 2) ** 2 / 2)

    X1 = np.linspace(0, 2 * np.pi * 3, 64)
    X2 = np.copy(X1)[::2]
    X1 += np.random.randn(len(X1)) * X1.ptp() / (len(X1) - 1) * 0.25
    X2 += np.random.randn(len(X2)) * X2.ptp() / (len(X2) - 1) * 0.25

    E1, E2 = [np.random.poisson(10, size=len(X)) * 0.005 for i, X in enumerate([X1, X2])]
    E2 *= 2

    lag_true = np.pi
    Y1 = f(X1) + np.random.randn(len(E1)) * E1
    Y2 = f(X2 + lag_true) + np.random.randn(len(E2)) * E2

    #::::::::::::::::::::
    # Lightcurve object
    data_1, data_2 = lightcurve(X1, Y1, E1), lightcurve(X2, Y2, E2)


    #::::::::::::::::::::
    # Make Litmus Object
    test_litmus = litmus(model = None,
                         fitproc= fitprocs.ICCF)

    test_litmus.add_lightcurve(data_1)
    test_litmus.add_lightcurve(data_2)

    print("Fitting Start")
    test_litmus.fit()
    print("Fitting complete")

    results = test_litmus.get_samples(N = 1_000)["lag"]

    print("Recovered lag is %.2f +/- %.2f, consistent with true of lag of %.2f at %.2f sigma" %(results.mean(), results.std(), lag_true, abs(lag_true-results.mean())/results.std()))





