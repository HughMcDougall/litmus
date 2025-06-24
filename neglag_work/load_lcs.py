import numpy as np
import pandas as pd
from glob import glob
import litmus

lc_names = []
['tau', 'b', 'z', 'amp', 'r', 'lag', 'lag_two']
_targs = glob("./lcs/*_cont.csv")
for targ in _targs:
    D1, D2 = np.loadtxt(targ, delimiter=","), np.loadtxt(targ.replace("cont", "resp"), delimiter=",")

    tau, b, z, amp, r, lag, lag_two = np.loadtxt(targ.replace("cont", "prms"), delimiter=",")[1:]
    logtau = np.log(tau)
    logamp = np.log(amp)

    params = {'lag': lag,
              'logtau': logtau,
              'lag_two': lag_two,
              'lag_relamp': r,
              }

    name = targ
    name = name.replace("./lcs/", "")
    name = name.replace("_cont.csv", "")

    lc_1 = litmus.lightcurve(*D1.T)
    lc_2 = litmus.lightcurve(*D2.T)

    #lc_1 = lc_1.normalize()
    #lc_2 = lc_2.normalize()


    exec("%s = (lc_1, lc_2, params)" % name)

    lc_names.append(name)
    del D1, D2, name, params, tau, b, z, amp, r, lag, lag_two, logtau, logamp
