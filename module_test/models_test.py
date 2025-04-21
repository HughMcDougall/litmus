# ============================================
# ============================================
# Testing
from litmus.models import *

if __name__ == "__main__":
    print("Testing models.py")

    from litmus.mocks import mock
    import matplotlib

    matplotlib.use('module://backend_interagg')

    print("Creating mocks...")
    mymock = mock(cadence=[7, 7])
    true_params = mymock.params()
    mymock.plot()

    print("Creating model...")
    my_model = GP_simple()
    my_model.debug = True
    lc_1, lc_2 = mymock.lc_1, mymock.lc_2
    data = my_model.lc_to_data(lc_1, lc_2)

    if False:
        print("Testing sampling and density...")
        prior_samps = my_model.prior_sample(num_samples=50_000)

        lag_samps = dict_extend(mymock.params(), {'lag': prior_samps['lag']})
        prior_LL = my_model.log_density(lag_samps, data=data)

        plt.scatter(lag_samps['lag'], prior_LL - prior_LL.max(), s=1, c='k')
        plt.axvline(true_params['lag'], ls='--', c='k')

        plt.grid()
        plt.xlim(*my_model.prior_ranges['lag'])
        plt.xlabel("Lag")
        plt.ylabel("Log Posterior (Arb Units)")
        plt.gcf().axes[0].set_yticklabels([])

        plt.show()

    # ----------------------------

    print("Testing find_seed...")
    seed_params, val_seed = my_model.find_seed(data=data)

    tol_seed = my_model.opt_tol(seed_params, data)
    print("Scan starting at %.2e sigma from optimum & log density %.2f" % (tol_seed, val_seed))

    print("Testing Scan...")
    scanned_params = my_model.scan(seed_params,
                                   data,
                                   optim_kwargs={'increase_factor': 1.1,
                                                 'max_stepsize': 0.2
                                                 },
                                   precondition='half-eig',
                                   optim_params=['lag', 'logtau']
                                   )

    val = my_model.log_density(scanned_params, data)
    tol = my_model.opt_tol(scanned_params, data)
    print("Scan settled at %.2e sigma from optimum & log density %.2f" % (tol, val))

    maxlen = max([len(key) for key in my_model.paramnames()])
    S = "%s \t Truth \t Seed \t MAP \n" % ("Param".ljust(maxlen))
    for key in my_model.paramnames():
        S += "%s \t %.2f \t %.2f \t %.2f \n" % (
            key.ljust(maxlen), mymock.params()[key], seed_params[key], scanned_params[key])
    print(S)

    print("All checks okay.")
