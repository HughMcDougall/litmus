'''
litmus.py

Contains the main litmus object class, which acts as a user-friendly interface with the models statistical models
and fitting procedure. In future versions, this will also give access to the GUI.

todo
    - This entire class to be re-done to take multiple models instead of multiple lightcurves
    - Possibly add hdf5 saving to chain output
    - Maybe add save_litmus() /w pickling?
    - Need to have better handling of the "fitting method inherit" feature, especially with refactor / redo
    -
'''

# ============================================
# IMPORTS
import sys
import csv
import pandas as pd

from chainconsumer import ChainConsumer, Chain, ChainConfig, PlotConfig

import matplotlib
import litmus._types as _types

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

from litmus.logging import logger


# =========================================================
# LITMUS (Fit Handler)
# =========================================================

class LITMUS(logger):
    """
    A front-facing UI class for interfacing with the fitting procedures.
    """

    def __init__(self, fitproc: fitting_procedure = None):

        logger.__init__(self)
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
        self.prior_samples = self.model.prior_sample(self.Nsamples)
        self.C = ChainConsumer()

        self.C.set_override(ChainConfig(smooth=0, linewidth=2, plot_cloud=True, shade_alpha=0.5))

        # self.C.add_chain(Chain(samples=DataFrame.from_dict(self.prior_samples), name="Prior", color='gray'))
        if self.fitproc.has_run:
            self.samples = self.fitproc.get_samples(self.Nsamples)
            self.samples = self.fitproc.get_samples(self.Nsamples)
            self.C.add_chain(Chain(samples=DataFrame.from_dict(self.samples), name="Lightcurves %i-%i"))
            self.msg_err("Warning! LITMUS object built on pre-run fitting_procedure. May have unexpected behaviour.")

        return

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "Nsamples" and hasattr(self, "samples") and self.samples != {}:
            super().__setattr__("samples", self.fitproc.get_samples(value))

    def add_lightcurve(self, lc: lightcurve):
        """
        Add a lightcurve 'lc' to the LITMUS object
        """
        self.lightcurves.append(lc)
        return

    def remove_lightcurve(self, i: int) -> None:
        """
        Remove lightcurve of index 'i' from the LITMUS object
        """
        N = len(self.lightcurves)

        if i < N:
            del self.lightcurves[i]
        else:
            self.msg_err("Tried to delete lightcurve %i but only have %i lightcurves. Skipping" % (i, N))
        return

    # ----------------------
    # Running / interface /w fitting methods
    def prefit(self, i=0, j=1):
        """
        Performs the full fit for the chosen stats model and fitting method.
        """

        lc_1, lc_2 = self.lightcurves[i], self.lightcurves[j]
        self.data = self.model.lc_to_data(lc_1, lc_2)

        self.fitproc.prefit(lc_1, lc_2)

    def fit(self, i=0, j=1) -> None:
        """
        Performs the full fit for the chosen stats model and fitting method.
        """

        lc_1, lc_2 = self.lightcurves[i], self.lightcurves[j]
        self.data = self.model.lc_to_data(lc_1, lc_2)

        self.fitproc.fit(lc_1, lc_2)

        self.samples = self.fitproc.get_samples(self.Nsamples)
        self.C.add_chain(Chain(samples=DataFrame.from_dict(self.samples), name="Lightcurves %i-%i" % (i, j)))

    def save_chain(self, path: str = None, headings: bool = True) -> None:
        """
        Saves the litmus's output chains to a .csv file at "path"
        If headings=True (default) then the names of the parameters will be written to the first row of the tile
        #todo - this needs updating
        """
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

    def read_chain(self, path: str, header: _types.Iterable[str] | None = None):
        """
        #todo needs updating
        """
        # Reading the CSV into a DataFrame
        df = pd.read_csv(path)

        if header is None:
            keys = df.columns
        else:
            keys = header.copy()

        # Converting DataFrame to dictionary of numpy arrays
        out = {col: df[col].to_numpy() for col in keys}

        if out.keys() <= set(self.fitproc.stat_model.paramnames()):
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

    def plot_lightcurves(self, model_no: int =0, Nsamples: int = 1, Tspan: None | list[float, float] = None, Nplot: int = 1024,
                         dir: str | None = None, show: bool = True) -> matplotlib.figure.Figure():
        """
        Plots the interpolated lightcurves for one of the fitted models
        :param model_no: Which model to plot the lightcurves for
        :parameter Nsamples: Number of posterior samples to draw from when plotting
        :parameter Tspan: Span of time values to plot over. If None, will use the max / min times of lc_1 and lc_2
        :parameter Nplot: Number of points in the interpolated lightcurve
        :parameter dir: If not None, will save to this filepath
        :parameter show: If True, will plt.show() the plot
        """

        self.msg_err("plot_lightcurve() not yet implemented")
        fig = plt.figure()
        return fig

    def plot_parameters(self, model_no: int | None = None, Nsamples: int = None, CC_kwargs: dict = {},
                        show: bool = True,
                        prior_extents: bool = False, dir: str | None = None) -> matplotlib.figure.Figure:
        """
        Creates a nicely formatted chainconsumer plot of the parameters
        :param model_no: Which model to plot the lightcurves for. If None, will plot for all
        :param Nsamples: Number of posterior samples to draw from when plotting
        :parameter CC_kwargs: Keyword arguments to pass to the chainconsumer constructor
        :parameter show: If True, will show the plot
        :parameter prior_extents: If True, will use the model prior range for the axes limits (Defaults to false if multiple models used)
        :parameter dir: If not None, will save to this filepath
        Returns the matplotlib figure

        # todo - refactor for multi-model implementation
        """

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
        params_toplot = [param for param in self.model.free_params() if self.samples[param].ptp() != 0]
        if len(params_toplot) == 0:
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

    def lag_plot(self, Nsamples: int = None, show: bool = True, extras: bool = True, prior_extents=False,
                 dir: str | None = None, ) -> matplotlib.figure.Figure:
        """
        Creates a nicely formatted chainconsumer plot of the marginalized lag plot
        :param Nsamples: Number of posterior samples to draw from when plotting
        :parameter show: If True, will show the plot
        :parameter extras: If True, will add any fitting method specific extras to the plot
        :parameter dir: If not None, will save to this filepath
        :parameter prior_extents: If True, will use the model prior range for the axes limits (Defaults to false if multiple models used)

        Returns the matplotlib figure
        """
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
        if prior_extents: fig.axes[0].set_xlim(*self.model.prior_ranges['lag'])
        fig.axes[0].set_ylim(*fig.axes[0].get_ylim())
        fig.tight_layout()

        fig.axes[0].grid()

        # Method specific plotting of fun stuff
        if extras:
            if isinstance(self.fitproc, fitting_methods.hessian_scan):
                X, logY = self.fitproc._get_slices('lags', 'logZ')

                if self.fitproc.interp_scale == 'linear':
                    Y = np.exp(logY - logY.max())
                    Y /= np.trapz(Y, X)
                    fig.axes[0].plot(X, Y)

                elif self.fitproc.interp_scale == 'log':
                    Xterp = np.linspace(*self.model.prior_ranges['lag'], self.Nsamples)
                    logYterp = np.interp(Xterp, X, logY, left=logY[0], right=logY[-1])
                    Yterp = np.exp(logYterp - logYterp.max())
                    Yterp /= np.trapz(Yterp, Xterp)
                    fig.axes[0].plot(Xterp, Yterp)

                plt.scatter(self.fitproc.lags, np.zeros_like(self.fitproc.lags), c='red', s=20)
                plt.scatter(X, np.zeros_like(X), c='black', s=20)
        if dir is not None:
            plt.savefig(dir)
        if show: fig.show()
        return (fig)

    def diagnostic_plots(self, dir: str | None = None, show: bool = False, **kwargs):
        """
        Generates a diagnostic plot window
        :param dir: If not None, will save to this filepath
        :param show: If True, will show the plot

        If dir!=None, will plt.savefig to the filepath 'dir' with **kwargs
        """
        if hasattr(self.fitproc, "diagnostics"):
            self.fitproc.diagnostics()
        else:
            self.msg_err("diagnostic_plots() not yet implemented for fitting method %s" % (self.fitproc.name))

        if dir is not None:
            plt.savefig(dir, **kwargs)

        if show: plt.show()

        return