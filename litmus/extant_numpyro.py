'''
A holder file for numpyro working, to be integrated into the models.py file
'''

# ============================================
# # NUMPYRO FUNCTIONS


# -----------------------

def GPjitter(T,Y,E,bands,
              basekernel = tinygp.kernels.quasisep.Exp,
              config_params = _default_config):
    '''
    :param T:
    :param Y:
    :param E:
    :param bands:
    :param basekernel:
    :param config_params:
    :return:
    '''

    # Sample distributions
    logtau   = numpyro.sample('logtau', dist.Uniform(config_params['logtau'][0], config_params['logtau'][1]))
    logamp   = numpyro.sample('logamp', dist.Uniform(config_params['logamp'][0], config_params['logamp'][1]))

    rel_amp  = numpyro.sample('rel_amp', dist.Uniform(0.0, config_params['rel_amp']))
    mean     = numpyro.sample('mean', dist.Uniform(config_params['mean'][0], config_params['mean'][1]))
    rel_mean = numpyro.sample('rel_mean', dist.Uniform(config_params['rel_mean'][0], config_params['rel_mean'][1]))
    jitter   = numpyro.sample('jitter', dist.Uniform(0.0, config_params['jitter']))

    lag = numpyro.sample('lag', dist.Uniform(config_params['lag'][0], config_params['lag'][1]))

    # Conversions to gp-friendly form
    amp, tau = jnp.exp(logamp), jnp.exp(logtau)

    diag = jnp.diag(jnp.square(E + bands * jitter * (amp * rel_amp)))

    delays = jnp.array([0, lag])
    amps   = jnp.array([amp,  rel_amp*amp])
    means  = jnp.array([mean, mean + rel_mean])

    # Build and sample GP
    gp = build_gp(T-delays[bands], Y, diag, bands, tau, amps, means, basekernel=basekernel)
    numpyro.sample("Y", gp.numpyro_dist(), obs = Y)

# -----------------------
GPsimple = handlers.substitute(GPjitter, {'jitter':0.0})

# -----------------------

def GPoutlier(T,Y,E,bands,
              basekernel = tinygp.kernels.quasisep.Exp,
              config_params = _default_config):
    '''
    :param T:
    :param Y:
    :param E:
    :param bands:
    :param basekernel:
    :param config_params:
    :return:
    '''

    # Sample distributions
    logtau   = numpyro.sample('logtau', dist.Uniform(config_params['logtau'][0], config_params['logtau'][1]))
    logamp   = numpyro.sample('logamp', dist.Uniform(config_params['logamp'][0], config_params['logamp'][1]))

    rel_amp  = numpyro.sample('rel_amp', dist.Uniform(0.0, config_params['rel_amp']))
    mean     = numpyro.sample('mean', dist.Uniform(config_params['mean'][0], config_params['mean'][1]))
    rel_mean = numpyro.sample('rel_mean', dist.Uniform(config_params['rel_mean'][0], config_params['rel_mean'][1]))

    lag = numpyro.sample('lag', dist.Uniform(config_params['lag'][0], config_params['lag'][1]))

    outlier_spread = numpyro.sample('outlier_spread', dist.Uniform(0.0, config_params['outlier_spread']), sample_shape=(2,))
    Q = numpyro.sample('outlifer_frac', dist.Uniform(0.0, config_params['outlier_frac']))

    # Conversions to gp-friendly form
    amp, tau = jnp.exp(logamp), jnp.exp(logtau)
    diag = jnp.diag(jnp.square(E))

    delays = jnp.array([0, lag])
    amps   = jnp.array([amp,  rel_amp*amp])
    means  = jnp.array([mean, mean + rel_mean])

    # Build and sample GP
    gp_fg = build_gp(T-delays[bands], Y, diag, bands, tau, amps, means, basekernel=basekernel)
    gp_bg = build_gp(T - delays[bands], Y, diag, bands, tau=0.0, amps = outlier_spread, means = means, basekernel=basekernel)

    mix_dist = MixtureGeneral(dist.Categorical(1.0-Q,Q),
                          [
                              gp_fg.numpyro_dist(),
                              gp_bg.numpyro_dist()
                          ])


    numpyro.sample("Y", mix_dist, obs = Y)
'''