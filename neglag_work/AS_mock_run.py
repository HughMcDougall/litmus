import litmus.fitting_methods
from neglag_model import model_twolag
from load_lcs import *

tests = [lc1_2, lc32801_8, lc33_2, lc348_2, lc363_2, lc38_2, lc5_5, lc898_8, lc9390_8]

lc_1, lc_2, true_params = lc1_2

model = model_twolag()
# model.set_priors(true_params)

fitter = litmus.fitting_methods.nested_sampling(model,
                                                num_live_points=model.dim()*8,
                                                verbose=10,
                                                warn=10,
                                                debug=10
                                                )
data = model.lc_to_data(lc_1, lc_2)
fitter.fit(lc_1, lc_2)
lt = litmus.LITMUS(fitter)
lt.plot_parameters(truth=true_params)
lt.save_chain("./out.csv")