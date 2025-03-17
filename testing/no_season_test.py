import numpy as np
import matplotlib.pyplot as plt
import litmus
from litmus.mocks import mock

# ----------------------------

my_mock = mock(maxtime=360 * 4, season=0, seasonvar=0, E=[0.1, 0.1], cadence=[1, 1], lag=106, tau=np.exp(5.6))
my_mock = my_mock(16)
my_mock.plot()

lc_1, lc_2 = my_mock.lc_1, my_mock.lc_2

my_model = litmus.models.GP_simple()

data = my_model.lc_to_data(lc_1, lc_2)

fitting_procedure = litmus.fitting_methods.hessian_scan(my_model,
                                                        Nlags = 32,
                                                        )