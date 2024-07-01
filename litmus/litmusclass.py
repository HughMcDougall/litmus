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


# =========================================================
# LITMUS (Fit Handler)
# =========================================================

class litmus(object):
    '''
    Wrapper for lag recovery methods and configuration
    TODO
    - parameter search ranges
    - add lightcurve
    - 'fit_lag()'
    - 'plot_lightcurves'
    '''

    def __init__(self, model=None, fitproc=None):
        self.model = model
        self.fitproc = fitproc

        self.lightcurves = {}

        self.main_stream = sys.stdout
        self.err_stream = sys.stderr

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
    from mocks import mock_A_01, mock_A_02, lag_true

    #::::::::::::::::::::
    # Make Litmus Object
    test_litmus = litmus(model=None,
                         fitproc=fitprocs.ICCF)

    test_litmus.add_lightcurve(mock_A_01)
    test_litmus.add_lightcurve(mock_A_02)

    print("Fitting Start")
    test_litmus.fit()
    print("Fitting complete")

    results = test_litmus.get_samples(N=1_000)["lag"]

    print("Recovered lag is %.2f +/- %.2f, consistent with true of lag of %.2f at %.2f sigma" % (
    results.mean(), results.std(), lag_true, abs(lag_true - results.mean()) / results.std()))
