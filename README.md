<p style="text-align:center;">

# LITMUS

<u>**L**ag **I**nference **T**hrough the **M**ixed **U**se of **S**amplers</u>

</p>

LITMUS is an in-progress program that uses modern statistical techniques, like
nested sampling and stochastic variational inference, in combination with
cutting edge programming tools like the just-in-time compilation framework `jax`
and its bayesian modelling package `NumPyro`, to perform the messy task of lag
recovery in AGN reverberation mapping.

![LITMUS](./logo.png)

This project is still very much in the early stages. If you have any 
questions, contact the author directly at hughmcdougallemail@gmail.com

## Installation

In it's current pre-release state, `LITMUS` is a bit fragile in it's  
installation thanks to making use of some under-developement software 
packages and the outdated `chainconsumer==0.3.4.0`. At time of writing 
(23/10/24) this is still changing.

### Simple Installation

First make sure you have a recent version of python running (`3.10`-`3.12`) 
and then install directly from the git repo:

```
pip install "git+https://github.com/HughMcDougall/litmus"
```

### Explicit Installation

If you find the above doesn't work, try first installing the dependencies one by
one, starting with the commonplace
python packages:

```
pip install numpy matplotlib scikit-learn
```

Then the `JAX` ecosystem and `numpyro` utilities:

```
pip install jax jaxlib jaxopt
pip install numpyro tinygp
```

For plotting utilties we need chainconsumer. The classic version is only
supported with old versions of scipy, which in turn are only useable on 
older version of python. As such, this step can be a little messy:

**If using python `3.8`-`3.10`:**

```
pip install scipy==1.11.4
pip install chainconsumer==0.34.0
```

**If using python `3.11`-`3.12`:**

```
pip install scipy
pip install chainconsumer
```

**Nested Sampling**  
Nested sampling is <u>currently incomplete (23/10/24)</u> but will make use of both 
jaxns and polychord If you want to make use of [`jaxns` nested sampling](https://github.com/Joshuaalbert/jaxns),  
you'll need to install it with:

```
pip install etils tensorflow_probability
pip install jaxns
```

If you would rather use the polychord sampler, you can attempt this by 
following the [documentation](https://github.com/PolyChord/PolyChordLite), 
or trying your luck with:

```
git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
make
pip install .
```

--------------
_Last Update 23/10_