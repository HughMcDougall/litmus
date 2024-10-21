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

import numpy as np
import jax.numpy as jnp

from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

import models
from models import stats_model
import fitting_methods
from fitting_methods import fitting_procedure
from lightcurve import lightcurve


# =========================================================
# LITMUS (Fit Handler)
# =========================================================

class LITMUS(object):
    '''
    A front-facing UI class for interfacing with the fitting procedures.
    '''

    def __init__(self, fitproc: fitting_procedure = None):

        # ----------------------------

        self.out_stream = sys.stdout
        self.err_stream = sys.stderr
        self.verbose = True
        self.debug = False

        # ----------------------------

        if fitproc is None:
            self.msg_err("Didn't set a fitting method, using GP_simple")
            self.model = models.GP_simple()

            self.msg_err("Didn't set a fitting method, using hessian scan")

            fitproc = fitting_methods.hessian_scan(stat_model=self.model)

        self.model = fitproc.stat_model
        self.fitproc = fitproc

        # ----------------------------
        self.lightcurves = []
        self.data = None

        self.Nsamples = 50_000
        self.samples = {}
        self.C = ChainConsumer()

        self.C.configure(smooth=0, summary=True, linewidths=2, cloud=True, shade_alpha=0.5)

        return

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "Nsamples" and hasattr(self, "samples") and self.samples != {}:
            super().__setattr__("samples", self.fitproc.get_samples(value))

    def add_lightcurve(self, lc: lightcurve):
        '''
        Add a lightcurve 'lc' to the LITMUS object
        '''
        self.lightcurves.append(lc)
        return

    def remove_lightcurve(self, i: int):
        '''
        Remove lightcurve of index 'i' from the LITMUS object
        '''
        N = len(self.lightcurves)
        if i < N:
            del self.lightcurves[i]
        else:
            self.msg_err("Tried to delete lightcurve %i but only have %i lightcurves. Skipping" % (i, N))
        return

    # ----------------------
    # Running / interface /w fitting methods
    def fit(self, i=0, j=1):
        '''
        Performs the full fit for the chosen stats model and fitting method.
        '''

        lc_1, lc_2 = self.lightcurves[i], self.lightcurves[j]
        self.data = self.model.lc_to_data(lc_1, lc_2)

        self.fitproc.fit(lc_1, lc_2)

        self.samples = self.fitproc.get_samples(self.Nsamples)
        self.C.add_chain(self.samples, name="Lightcurves %i-%i" % (i, j))

    # ----------------------
    # Plotting

    def plot_lightcurves(self):
        self.msg_err("plot_lightcurve() not yet implemented")
        return

    def plot_parameters(self, Nsamples: int = None, CC_kwargs={}, show=True, prior_extents=False):
        '''
        Creates a nicely formatted chainconsumer plot of the parameters
        Returns the chainconsumer plot figure
        '''
        if Nsamples is not None and Nsamples != self.Nsamples:
            C = ChainConsumer()
            samps = self.fitproc.get_samples(Nsamples, **CC_kwargs)
            C.add_chain(samps)
        else:
            C = self.C

        fig = C.plotter.plot(parameters=self.model.free_params(),
                             extents=self.model.prior_ranges if prior_extents else None,
                             **CC_kwargs)
        fig.tight_layout()
        if show: fig.show()

        return fig

    def lag_plot(self, Nsamples: int = None, show=True, extras=True):
        '''
        Creates a nicely formatted chainconsumer plot of the parameters
        Returns the ChainConsumer object
        '''
        if 'lag' not in self.model.free_params():
            return

        if Nsamples is not None and Nsamples != self.Nsamples:
            C = ChainConsumer()
            samps = self.fitproc.get_samples(Nsamples)
            C.add_chain(samps)
        else:
            C = self.C
        fig = C.plotter.plot_distributions(extents=self.model.prior_ranges, parameters=['lag'], figsize=(8, 4))
        fig.axes[0].set_ylim(*fig.axes[0].get_ylim())
        fig.tight_layout()

        fig.axes[0].grid()

        if extras:

            if isinstance(self.fitproc, fitting_methods.hessian_scan):
                X, Y = self.fitproc.scan_peaks['lag'], self.fitproc.log_evidences
                Y -= Y.max()
                Y = np.exp(Y)
                Y /= Y.sum()
                fig.axes[0].plot(X, Y)

                plt.scatter(self.fitproc.lags, np.zeros_like(self.fitproc.lags), c='red', s=20)
                plt.scatter(X, np.zeros_like(X), c='black', s=20)

        if show: fig.show()
        return (fig)

    def diagnostic_plots(self):
        try:
            self.fitproc.diagnostics()
        except:
            self.msg_err("diagnostic_plots() not yet implemented for fitting method %s" % (self.fitproc.name))
        return

    # ----------------------
    # Error message printing
    def msg_err(self, *x: str, end='\n', delim=' '):
        '''
        Messages for when something has broken or been called incorrectly
        '''
        if True:
            for a in x:
                print(a, file=self.err_stream, end=delim)

        print(end, end='')
        return

    def msg_run(self, *x: str, end='\n', delim=' '):
        '''
        Standard messages about when things are running
        '''
        if self.verbose:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return

    def msg_verbose(self, *x: str, end='\n', delim=' '):
        '''
        Explicit messages to help debug when things are behaving strangely
        '''
        if self.debug:
            for a in x:
                print(a, file=self.out_stream, end=delim)

        print(end, end='')
        return


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    #::::::::::::::::::::
    # Mock Data
    from mocks import *

    # mymock = mymock.copy(E=[0.05,0.1], lag=300)
    mymock = mock(cadence=[7, 30], E=[0.15, 0.5], season=180, lag=256, tau=200.0)
    # mymock=mock_B
    mymock.plot()
    mymock(10)
    plt.show()

    lag_true = mymock.lag

    test_model = models.GP_simple()
    test_model.set_priors({'mean': [0.0, 0.0]})

    seed_params = mymock.params()

    #::::::::::::::::::::
    # Make Litmus Object
    fitting_method = fitting_methods.SVI_scan(stat_model=test_model,
                                              Nlags=128,
                                              init_samples=5_000,
                                              grid_bunching=0.6,
                                              optimizer_args={'tol': 1E-3,
                                                              'maxiter': 512,
                                                              'increase_factor': 1.2,
                                                              },
                                              reverse=False,

                                              seed_params=seed_params,
                                              debug=True

                                              )

    test_litmus = LITMUS(fitting_method
                         )

    test_litmus.add_lightcurve(mymock.lc_1)
    test_litmus.add_lightcurve(mymock.lc_2)

    print("Fitting Start")
    test_litmus.fit()
    print("Fitting complete")

    test_litmus.plot_parameters()
    test_litmus.lag_plot()
    test_litmus.diagnostic_plots()
