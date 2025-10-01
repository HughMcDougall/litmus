from tinygp.helpers import JAXArray

import numpy as np
from litmus_rm import *

from tinygp import GaussianProcess
import tinygp
from litmus_rm import _types

import jax.numpy as jnp
import jax


# ----------------------------------------------
# Likelihood function working
def mean_func(means, Y) -> _types.ArrayN:
    """
    DEPRECATED - means are subtracted in the model now
    Utitlity function to take array of constants and return as gp-friendly functions
    """
    t, band = Y
    return (means[band])


@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    """
    Multi-band GP kernel that knows how to scale GP to output amplitudes
    """
    scale: JAXArray | float
    amplitudes: jnp.ndarray
    lag: float

    def coord_to_sortable(self, Y) -> float:
        """
        Extracts the time value from the (time,band) coordinate so the GP can interpret the ordering of points
        in multiple bands
        """
        t, band = Y
        return t

    def observation_model(self, Y) -> float:
        """
        Scales the prediction for each band by their respective band amplitude in the predicted model
        """
        t, band = Y
        return self.amplitudes[band] * self.kernel.observation_model(t)

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        T1, B1 = X1
        T2, B2 = X2
        T_1 = T1 - B1 * self.lag
        T_2 = T1 - B2 * self.lag
        out = self.kernel.transition_matrix(self.kernel.coord_to_sortable(T_1), self.kernel.coord_to_sortable(T_2))
        return out

# ----------------------------------

T = jnp.linspace(0, 5000, 512 * 4)
Y = np.random.randn(512 * 4)
Y = jnp.array(Y)
diag = jnp.array(np.random.rand(512 * 4)) * 0
tau = 300.0
bands = jnp.where(np.arange(512 * 4) % 2 == 0, 0, 1)

amps = jnp.array([1.0, 1.0])
means = jnp.array([0.0, 0.0])
basekernel = tinygp.kernels.quasisep.Exp

K1 = Multiband(
    kernel=basekernel(scale=tau),
    amplitudes=jnp.array([0.0, 0.0]),
    lag=0.0,
    scale=tau
)

K2 = Multiband(
    kernel=basekernel(scale=tau),
    amplitudes=jnp.array([1.0, 1.0]),
    lag=00.0,
    scale=tau
)

lag0 = 0
Tsort = T + bands * lag0
Isort = Tsort.argsort()
Tsort, bands = Tsort[Isort], bands[Isort]

K = K1 + K2
# Mean functions for offsetting signals
meanf = lambda X: mean_func(means, X)

# Construct GP object and return
gp = GaussianProcess(
    K,
    (Tsort, bands),
    diag=diag,
    mean=0.0,
)

Ypred = gp.sample(key=jax.random.PRNGKey(1))

for i in (0, 1):
    Igood = np.isnan(Ypred)
    Tplot = Tsort - bands * lag0
    plt.plot(Tplot[bands == i], Ypred[bands == i])
plt.show()
