# LITMUS todo

## MAJOR ISSUES

## Major Features to Add
1. GUI interface for LITMUS object (?)
2. Full documentation / docstrings
3. Clean up example & test folder to be clearer
4. Add lightcurve reconstruction method to models

### Minor Additions / corrections
1. Add information estimator to hessian + SVI scan
2. Various optimization / parallelization fixes in hessian scan
3. Clean up presentation of the diagnostics in the SVI and hessian scan
4. Add re-fit to SVI scan
5. Create non-bayesian subclass of `fitting_method()` that doesn't require a model intake in `__init__()`

### Possible Optional Additions
1. Implement the stochastic optimization method as a precursor to ISthMUS
2. Add a scan + HMC method
3. Add Von-Nuemann scanner to complement ICCF
