import litmus_rm.fitting_methods
from litmus_rm.mocks import mock
import numpy as np

import pickle

import os

model = litmus_rm.models.GP_simple()
null = litmus_rm.models.GP_simple_null()
noise = litmus_rm.models.whitenoise_null()

fitter_model = litmus_rm.fitting_methods.hessian_scan(model, verbose=5, LL_threshold = 200)
fitter_null = litmus_rm.fitting_methods.hessian_scan(null, verbose=5, LL_threshold = 200)
# fitter_noise = litmus.fitting_methods.hessian_scan(noise)

for seed in range(256):
    if os.path.isfile('%smockresults-%i.pckl' % ("./", seed)): continue

    import time

    np.random.seed(seed)
    tau = np.exp(np.random.rand() * (10 - 4.6) + 4.6)
    lag = np.random.rand() * 1_000

    test_mock = mock(tau=tau, E=[0.015, 0.15], Evar = [0.0025, 0.025], lag=lag, seed=seed)

    print("For this run, model params are:")
    for (a, v) in test_mock.params().items():
        print("%s:\t%.2f" % (a, v))

    start_time = time.time()  # Record the start time

    for fitter in [fitter_model, fitter_null]:
        try:
            fitter.prefit(*test_mock.lcs())
            fitter.fit(*test_mock.lcs())

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            out = {"lag_posterior": fitter_model.get_samples(10_000)["lag"],
                   "bayes_factor": fitter_model.get_evidence(return_type="log")[0] - fitter_null.get_evidence(return_type="log")[0],
                   "mock_params": test_mock.params(),
                   "mock_data": [x._data for x in test_mock.lcs()],
                   "runtime": elapsed_time,
                   "estmaps": fitter.estmap_params
                   }


        except:
            out = {"lag_posterior": np.zeros(10_000),
                   "bayes_factor": np.nan,
                   "mock_params": test_mock.params(),
                   "mock_data": [x._data for x in test_mock.lcs()],
                   "runtime": np.nan,
                   "estmaps": fitter.estmap_params
                   }

        with open('%smockresults-%i.pckl' % ("./", seed), 'wb') as outp:
            pickle.dump(out, outp, pickle.HIGHEST_PROTOCOL)
        del out

