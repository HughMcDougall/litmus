from litmus import *
from litmus._utils import dict_extend

mock = mocks.mock(lag=180, E = [0.1, 0.25], tau = 200)
mock = mock(2)
#mock.plot()
lc_1, lc_2 = mock.lc_1, mock.lc_2


model = models.GP_simple()
model.set_priors({
    'lag': [0, 800],
    'mean': [0.00, 0.0],
    'rel_mean': [0.0, 0.0],
    'logamp': [0.0, 0.0],
    'rel_amp': [1.00, 1.00],
})
data = model.lc_to_data(lc_1, lc_2)

if False:
    lagplot = np.linspace(*model.prior_ranges['lag'], 1024)
    LLs = model.log_density(dict_extend(mock.params(), {'lag':lagplot}), data)
    plt.plot(lagplot, LLs)
    plt.show()

meth = fitting_methods.JAVELIKE(model,
                                verbose=True,
                                debug=True,
                                num_warmup = 5_000,
                                num_samples = 100_000//512,
                                num_chains = 512
                                )

meth_2 = fitting_methods.hessian_scan(model,
                                      verbose=True,
                                      debug=True,
                                      Nlags = 64,
                                      precondition = "half-eig"
                                      )

for method in [meth, meth_2]:
    print(method.name)
    method.prefit(lc_1, lc_2)
    method.fit(lc_1, lc_2)
    samps = meth.get_samples()

    lt = LITMUS(method)
    lt.lag_plot()
    lt.plot_parameters()
