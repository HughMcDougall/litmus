'''
litmus.py

Contains the main litmus object class, which acts as a user-friendly interface with the models statistical models
and fitting procedure. In future versions, this will also give access to the GUI.

'''
# ============================================
# IMPORTS
import sys
import csv
import pandas as pd

from chainconsumer import ChainConsumer, Chain, ChainConfig, PlotConfig

from pandas import DataFrame

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

import litmus.models as models
from litmus.models import stats_model
import litmus.fitting_methods as fitting_methods
from litmus.fitting_methods import fitting_procedure
from litmus.lightcurve import lightcurve
from litmus._utils import *


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

        self.C.set_override(ChainConfig(smooth=0, linewidth=2, plot_cloud=True, shade_alpha=0.5))

        if self.fitproc.has_run:
            self.samples = self.fitproc.get_samples(self.Nsamples)
            self.C.add_chain(Chain(samples=DataFrame.from_dict(self.samples), name="Lightcurves %i-%i"))
            self.msg_err("Warning! LITMUS object built on pre-run fitting_procedure. May have unexpected behaviour.")

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
    def prefit(self, i=0, j=1):
        '''
        Performs the full fit for the chosen stats model and fitting method.
        '''

        lc_1, lc_2 = self.lightcurves[i], self.lightcurves[j]
        self.data = self.model.lc_to_data(lc_1, lc_2)

        self.fitproc.prefit(lc_1, lc_2)

    def fit(self, i=0, j=1):
        '''
        Performs the full fit for the chosen stats model and fitting method.
        '''

        lc_1, lc_2 = self.lightcurves[i], self.lightcurves[j]
        self.data = self.model.lc_to_data(lc_1, lc_2)

        self.fitproc.fit(lc_1, lc_2)

        self.samples = self.fitproc.get_samples(self.Nsamples)
        self.C.add_chain(Chain(samples=DataFrame.from_dict(self.samples), name="Lightcurves %i-%i" % (i, j)))

    def save_chain(self, path=None, headings=True):

        '''
        methods = ["numpy"]

        if method not in methods:
            err_msg = "Tried to use save_chain() with bad methd %s. Allowable methods are:" %method
            for method in methods: err_msg +="%s, " %x
            self.msg_err(err_msg)
        '''

        if path is None:
            path = "./%s_%s.csv" % (self.model.name, self.fitproc.name)
            if path[-4:] != ".csv": path += ".csv"

        rows = zip(*self.samples.values())
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            if headings: writer.writerow(self.samples.keys())
            # Write rows
            writer.writerows(rows)

    def read_chain(self, path, header = None):
        # Reading the CSV into a DataFrame
        df = pd.read_csv(path)

        if header is None:
            keys = df.columns
        else:
            keys = header.copy()

        # Converting DataFrame to dictionary of numpy arrays
        out = {col: df[col].to_numpy() for col in keys}

        if out.keys()<=set(self.fitproc.stat_model.paramnames()):
            self.samples = out
            self.msg_run("Loaded chain /w headings", *keys)
        else:
            self.msg_err("Tried to load chain with different parameter names to model")

    def config(self, **kwargs):
        '''
        Quick and easy way to pass arguments to the chainconsumer object.
        Allows editing while prote
        '''
        self.C.set_override(ChainConfig(**kwargs))
    # ----------------------

    # Plotting

    def plot_lightcurves(self):
        self.msg_err("plot_lightcurve() not yet implemented")
        return

    def plot_parameters(self, Nsamples: int = None, CC_kwargs={}, show=True, prior_extents=False, dir=None):
        '''
        Creates a nicely formatted chainconsumer plot of the parameters
        Returns the chainconsumer plot figure
        '''

        if Nsamples is not None and Nsamples != self.Nsamples:
            C = ChainConsumer()
            samples = self.fitproc.get_samples(Nsamples, **CC_kwargs)
            C.add_chain(Chain(samples=DataFrame.from_dict(samples), name='samples'))
        else:
            C = self.C

        if prior_extents:
            _config = PlotConfig(extents=self.model.prior_ranges, summarise=True,
                                 **CC_kwargs)
        else:
            _config = PlotConfig(summarise=True,
                                 **CC_kwargs)
        C.plotter.set_config(_config)
        params_toplot = [param for param in self.model.free_params() if self.samples[param].ptp()!=0]
        if len(params_toplot) ==0:
            fig = plt.figure()
            if show: plt.show()
            return fig

        try:
            fig = C.plotter.plot(columns=params_toplot,
                                 )
        except:
            fig = plt.figure()
            fig.text(0.5, 0.5, "Something wrong with plotter")
        fig.tight_layout()
        if show: fig.show()

        if dir is not None:
            plt.savefig(dir)

        return fig

    def lag_plot(self, Nsamples: int = None, show=True, extras=True, dir=None):
        '''
        Creates a nicely formatted chainconsumer plot of the parameters
        Returns the ChainConsumer object
        '''
        if 'lag' not in self.model.free_params():
            self.msg_err("Can't plot lags for a model without lags.")
            return

        if Nsamples is not None and Nsamples != self.Nsamples:
            C = ChainConsumer()
            samples = self.fitproc.get_samples(Nsamples)
            C.add_chain(Chain(samples=DataFrame.from_dict(samples), name="lags"))
        else:
            C = self.C

        _config = PlotConfig(extents=self.model.prior_ranges, summarise=True)
        C.plotter.set_config(_config)
        fig = C.plotter.plot_distributions(columns=['lag'], figsize=(8, 4))
        fig.axes[0].set_ylim(*fig.axes[0].get_ylim())
        fig.tight_layout()

        fig.axes[0].grid()

        if extras:

            if isinstance(self.fitproc, fitting_methods.hessian_scan):
                X, Y = self.fitproc.scan_peaks['lag'], self.fitproc.log_evidences
                Y -= Y.max()
                Y = np.exp(Y)
                Y /= np.trapz(Y, X)
                fig.axes[0].plot(X, Y)

                plt.scatter(self.fitproc.lags, np.zeros_like(self.fitproc.lags), c='red', s=20)
                plt.scatter(X, np.zeros_like(X), c='black', s=20)
        if dir is not None:
            plt.savefig(dir)
        if show: fig.show()
        return (fig)

    def diagnostic_plots(self, dir=None):
        if hasattr(self.fitproc, "diagnostics"):
            self.fitproc.diagnostics()
        else:
            self.msg_err("diagnostic_plots() not yet implemented for fitting method %s" % (self.fitproc.name))

        if dir is not None:
            plt.savefig(dir)

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

    import matplotlib

    matplotlib.use('module://backend_interagg')

    #::::::::::::::::::::
    # Mock Data
    from mocks import *

    # mymock = mymock.copy(E=[0.05,0.1], lag=300)
    mymock = mock(cadence=[7, 30], E=[0.15, 1.0], season=180, lag=180 * 3, tau=200.0)
    # mymock=mock_B
    mymock.plot()
    mymock(10)
    plt.show()

    lag_true = mymock.lag

    test_model = models.GP_simple()

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
