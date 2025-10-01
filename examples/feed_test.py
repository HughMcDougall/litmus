"""
A quick test .py to debug the feeding of test lags to a scan fitter
"""
import litmus_rm.models
from litmus_rm import *
import matplotlib.pyplot as plt
import numpy as np
from litmus_rm._utils import dict_extend

mock = mocks.mock_B
mock.plot()
model_1 = models.GP_simple(verbose=200, debug=False)
data = model_1.lc_to_data(mock.lc_1, mock.lc_2)

model_1.find_seed(data)
fitter_2 = fitting_methods.ICCF(model_1, verbose=5, debug=False, Nboot=64)
fitter_2.fit(mock.lc_1, mock.lc_2)
test_lags = fitter_2.get_samples(32)
fitter_3 = fitting_methods.hessian_scan(model_1, precondition="eig", verbose=3, debug=False,
                                        seed_params={"lag": fitter_2.get_peaks()['lag'][0]},
                                        #test_lags=fitter_2.get_samples(64)["lag"],
                                        #LL_threshold = 1000.0,
                                        warn = 10
                                        )
fitter_3.fit(mock.lc_1, mock.lc_2)
