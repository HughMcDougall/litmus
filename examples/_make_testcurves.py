"""
This is the file to generate the lightcurves in basic_fitting
"""

import numpy as np

from litmus import *

mock = litmus.mocks.mock(seed=3,
                         tau=600,
                         lag=300
                         )

mock.plot()

lc_1, lc_2 = mock.lcs()

np.savetxt("lc_1.csv", lc_1._data, delimiter=",")
np.savetxt("lc_2.csv", lc_2._data, delimiter=",")
