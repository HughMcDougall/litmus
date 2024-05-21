'''
lightcurve.py

End-user friendly classes and methods

HM 2024
'''

# ============================================
# IMPORTS

from dataclasses import dataclass
from copy import deepcopy as copy
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


# ============================================
# LIGHT CURVE
# ============================================

class lightcurve(dict):
    '''
    A wrapper class for lightcurves. Construct /w array-like inputs for time, signal and error (optional)
    like:   lightcurve(T,Y,E)

    Supports array-like addition and float-like addition / multiplication

    '''

    # ----------

    def __init__(self, T, Y, E=None):
        self.T = np.array(T, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        if E is None:
            self.E = np.zeros_like(T)
        else:
            self.E = E

        self._data = np.vstack(self.values()).T

        self._norm_mean, self._norm_amp = 0.0, 1.0
        self.normalized = False

    # ----------
    # Array-like

    def __len__(self):
        return (len(self.T))

    def __getitem__(self, key):

        if isinstance(key, slice):
            return (self._data[key, :])

        elif isinstance(key, int):
            return (self._data[key, :])

        else:
            if key == "T":
                return (self.T)
            elif key == "Y":
                return (self.Y)
            elif key == "E":
                return (self.E)
            else:
                return (None)

    def __str__(self):
        return (self._data.__str__())

    def __repr__(self):
        return ("Lightcurve len %i" % len(self))

    def __add__(self, other):
        try:
            add = float(other)
        except:
            add = np.array(T, dtype=np.float64)
            assert len(self) == len(add), "cannot add array-like object of different length to lightcurve"

        out = copy(self)
        out.Y += add
        out.normalized = False
        return (out)

    def __sub__(self, other):
        return (self + other * -1)

    def __mul__(self, other):
        try:
            mult = float(other)
        except:
            mult = np.array(T, dtype=np.float64)
            assert len(self) == len(mult), "cannot add array-like object of different length to lightcurve"

        out = copy(self)
        out.Y *= mult
        out.E *= mult
        out.normalized = False
        return (out)

    def __truediv__(self, other):
        return (self * (1 / other))

    def __abs__(self):
        out = copy(self)
        out.Y = abs(self.Y)
        return (out)

    # ----------
    # Dict-Like

    def keys(self):
        return (["T", "Y", "E"])

    def values(self):
        return ([self[key] for key in self.keys()])

    # ----------

    def __setattr__(self, key, value):
        if key in self.keys() or key in ["_data", "_norm_mean", "_norm_amp", "normalized"]:
            super(lightcurve, self).__setattr__(key, value)

            if False not in [hasattr(self, test) for test in self.keys()]:
                if key in self.keys():
                    self._data = np.vstack(self.values()).T
        else:
            raise Warning("Tried to set nonexistant lightcurve attribute %s" % key)

    def normalize(self):
        '''
        Esimates the mean and amplitude of the lighturve assuming uncorrelated measurements
        Returns a lightcurve object with this normalization
        '''

        if self.normalized: return (self)

        # Check if have errorbars
        no_errs = False
        E = self.E
        if max(E == 0):
            no_errs = True
            E = np.ones_like(self.E)

        # Get initial guess assuming no scatter
        w = E ** -2
        mean0 = np.average(self.Y, weights=w)
        var0 = np.average((self.Y - mean0) ** 2, weights=w)

        if no_errs:
            meanbest, varbest = mean0, var0
        else:
            L = lambda X: ((self.Y - X[0]) ** 2 / (self.E ** 2 + X[1]) + np.log(self.E ** 2 + X[1])).sum()
            meanbest, varbest = optimize.minimize(L, np.array([mean0, var0]), method='Nelder-Mead').x

        # Make and return copy
        out = copy(self)
        out -= meanbest
        out /= np.sqrt(varbest)

        # If any errors, revert to simple estiamte
        if np.any(np.isnan(out._data)):
            meanbest, varbest = mean0, var0
            out = copy(self)
            out -= meanbest
            out /= np.sqrt(varbest)

        # Store normalizing values for later
        out._norm_mean = meanbest
        out._norm_amp = np.sqrt(varbest)

        out.normalized = True

        return (out)

    def unnormalize(self):
        '''
        Reverses the effects of lightcurve.normalize().
        Returns a lightcurve object with mean and amplitude prior to normalize()
        '''
        out = copy(self)
        out *= self._norm_amp
        out += self._norm_mean
        out._norm_amp = 1.0
        out._norm_mean = 0.0
        out.normalized = False
        return (out)

    def trim(self, Tmin=None, Tmax=None):
        '''
        Returns a copy sub-sampled to only datapoints in the domain T in [Tmin,Tmax]
        '''
        if Tmin is None: Tmin = self.T.min()
        if Tmax is None: Tmax = self.T.max()
        I = np.where((self.T > Tmin) * (self.T < Tmax))[0]

        return (lightcurve(T=self.T[I],
                           Y=self.Y[I],
                           E=self.E[I]
                           ))

    def __getattr__(self, item):
        if item == "N":
            return(self.T.size)
        else:
            super().__getattribute__(item)


# =========================================================
# TESTING
if __name__ == "__main__":
    print(":)")
    T = np.linspace(0, 4 * np.pi, 128)
    Y = np.sin(T) * np.sqrt(2)
    E = (np.random.poisson(100, size=len(T)) / 100) * 1

    lc = lightcurve(T, Y + np.random.randn(len(T)) * E, abs(E))
    lc *= 1000
    lc += 1E3

    lc_calib = lc.normalize()
    lc_uncalib = lc_calib.unnormalize()

    fig, (a1, a2) = plt.subplots(2, 1)

    a1.errorbar(lc.T, lc.Y, lc.E, fmt='none', capsize=3)
    a1.errorbar(lc_uncalib.T, lc_uncalib.Y, lc_uncalib.E, fmt='none', capsize=1, c='r')

    a2.plot(T, Y)
    a2.errorbar(lc_calib.T, lc_calib.Y, lc_calib.E, fmt='none', capsize=2)
    plt.show()
