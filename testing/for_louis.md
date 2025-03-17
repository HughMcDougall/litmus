# The Problem

We have some function:

```
f(p|d)
```

Where:
* `p` is a keyed dictionary of floats
* `d` is a data object of indeterminant type 

Let `p={t,x,z}` where:
* `z` is fixed for all itterations
* `t` is itterated over `t_i in [t0,t1,t2...]`, `t_j<t_(j+1)`
* `x` is the set of parameters that we want to optimize at each itteration

We then want to get, for each `t_i`:
* Optimized parameters `x^_i`
* The gradient against `x` at these parameters, `df/dp (x^_i|t_i,d)`
* The hessian with respect to `x` at this same point

Where we can assume that all of the above are reasonably smooth functions of `t`.

## The Current Approach

1. Build a lambda function `f(x|t_i,d)`
2. Optimize /w jaxopt `BFGS`
3. Use `x^_i` as the starting point for optimization `t_i+1`