_HUGH MCDOUGALL 2024_

-----

-----

# LITMUS

<u>**L**ag **I**nference **T**hrough the **M**ixed **U**se of **S**amplers</u>

LITMUS is an in-progress program that uses modern statistical techniques, like nested sampling and stochastic variational inference, in combination with cutting edge programming tools like the just-in-time compilation framework
`jax` and its bayesian modelling package
`NumPyro`, to perform the messy task of lag recovery in AGN reverberation mapping.

![LITMUS](./logo.png)

This project is still very much in the early stages. If you have any questions, contact the author directly at [hughmcdougallemail@gmail.com](mailto:hughmcdougallemail@gmail.com).

-----

## Installation

### Simple Installation

First make sure you have a recent version of python running (`3.10`-`3.12`)
and then install directly from the git repo:

```
pip install "git+https://github.com/HughMcDougall/litmus"
```

### Explicit Installation

If you find the above doesn't work, try first installing the dependencies one by
one, starting with the commonplace python packages:

```
pip install numpy matplotlib scikit-learn
```

Then the `JAX` ecosystem and `numpyro` utilities:

```
pip install jax jaxlib jaxopt
pip install numpyro tinygp
```

For plotting utilties we need chainconsumer, which needs a newer version of scipy, which in turn requires a newish version of python.

**Requires Using python `3.11`-`3.12`:**

```
pip install scipy
pip install chainconsumer
```

**Nested Sampling**  
If you want to make use of [`jaxns` nested sampling](https://github.com/Joshuaalbert/jaxns), you'll need to install it with:

```
pip install etils tensorflow_probability
pip install jaxns
```

_Note: You have bump into some trouble installing `tensorflow_probability` if you don't have [`cmake`](https://cmake.org/) installed._

[//]: # (If you would rather use the polychord sampler, you can attempt this by)
[//]: # (following the [documentation]&#40;https://github.com/PolyChord/PolyChordLite&#41;,  or trying your luck with:)
[//]: # ()
[//]: # (```)
[//]: # (git clone https://github.com/PolyChord/PolyChordLite.git)
[//]: # (cd PolyChordLite)
[//]: # (make)
[//]: # (pip install .)
[//]: #```

-----

## Usage

### First Timers

```
import numpy as np
import matplotlib.pyplot as plt

import litmus
```

```
mymock = litmus.mocks.mock(3)
lc_1, lc_2 = mymock.lc_1, mymock.lc_2
mymock.plot()
```

Now, choose a model and set its priors(at time of writing only `GP_simple`,
which models both lightcurves as scaled and shifted damped random walks, is
implemented). For example suppose we know want to narrow our lag search
range to `[0,100] days`, and know that the lightcurves are normalized to
have zero mean:

```
my_model = litmus.models.GP_simple()
my_model.set_priors(
    {
    'lag': [0,100]
    'mean': [0,0]
    'rel_mean': [0,0]
    }
)
```

Now we choose a fitting method and tune it accordingly.

```
fitting_method = litmus.fitting_methods.hessian_scan(stat_model=my_model,
                                                  Nlags=32,
                                                  init_samples=5_000,
                                                  grid_bunching=0.8,
                                                  optimizer_args={'tol': 1E-3,
                                                                  'maxiter': 256,
                                                                  'increase_factor': 1.1,
                                                                  },
                                                  optimizer_args_init={'tol': 1E-4,
                                                                  'maxiter': 1024,
                                                                  'increase_factor': 1.01,
                                                                  },
                                                  reverse=False,
                                                  debug=False
                                                  )
```

Finally, wrap this in a `LITMUS` object, adding the lightcurves to it, which
makes running and getting results out as simple as a single line of code:

```
litmus_handler = litmus.LITMUS(fitting_method)

litmus_handler.add_lightcurve(lc_1)
litmus_handler.add_lightcurve(lc_2)
```

Now, fire off the fitting procedure:

```
litmus_handler.fit()
```

And finally plot the model parameters to see how we did:

```
litmus_handler.lag_plot()
litmus_handler.plot_parameters()
litmus_handler.plot_diagnostics()
```

-----

-----


_Last Update 23/10_