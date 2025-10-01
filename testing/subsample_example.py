'''
Demonstration of how the lag posterior smooths with bootstrapping
'''
import numpy as np

from litmus_rm import *
from litmus_rm import _utils


mock = mocks.mock(lag=200)
mock.plot()

model = models.GP_simple()

lc_1_0, lc_2_0 = mock.lc_1, mock.lc_2
data_0 = model.lc_to_data(lc_1_0, lc_2_0)

lagplot = np.linspace(0, 1000, 1024, endpoint=False)

print("Doing first run")
LL0 = model.log_density(_utils.dict_extend(mock.params(), {'lag': lagplot}), data_0)

M = 256
LLs = np.zeros([M, len(lagplot)])
frac = np.exp(-1)
for i in range(M):
    if i % (256//10) == 0: print("%.1f %%" %(i/M*100), end='\t')
    I1 = np.random.choice(np.arange(0, len(lc_1_0)), int(frac * len(lc_1_0)), replace=True)
    I2 = np.random.choice(np.arange(0, len(lc_2_0)), int(frac * len(lc_2_0)), replace=True)

    T1, Y1, E1 = lc_1_0.T[I1], lc_1_0.Y[I1], lc_1_0.E[I1]
    T2, Y2, E2 = lc_2_0.T[I2], lc_2_0.Y[I2], lc_2_0.E[I2]

    Y1+=E1*np.random.normal(int(frac * len(lc_1_0)))
    Y2+=E2*np.random.normal(int(frac * len(lc_2_0)))

    lc_1 = lightcurve(T1, Y1, E1)
    lc_2 = lightcurve(T2, Y2, E2)

    data = model.lc_to_data(lc_1, lc_2)

    LL = model.log_density(_utils.dict_extend(mock.params(), {'lag': lagplot}), data)

    LLs[i, :] = LL

#---------------------------
# Plotting


med = np.median(LLs, axis=0)
I = np.random.choice(range(M),4,replace=False)


f, (a1,a2) = plt.subplots(2,1, sharex=True, figsize=(12,6))

a1.plot(lagplot, LL0, label = "True Posterior")
a1.set_ylim(a1.get_ylim()[0], 300.0)


a2.plot(lagplot, med, label = "Median After Sub-Sampling")
a2.fill_between(lagplot, np.percentile(LLs, 16, axis=0), np.percentile(LLs, 84, axis=0), zorder = -1, alpha=0.3, label = "1 $\sigma$ percentiles")
a2.set_ylim(med.min()*1.1, med.max()+med.ptp()*0.6)


label = "Individual Realizations"
for i in I:
    a2.plot(lagplot, LLs[i,:], alpha=0.5, lw=1, c='tab:blue', label = label)
    label = None

a1.scatter(lagplot[LL0.argmax()], LL0.max(), c='r', label = "Max Posterior Probability")
a2.scatter(lagplot[med.argmax()], med.max(), c='r')

f.supxlabel("$\Delta t$")
f.supylabel("Posterior Log-Density")

for a in [a1,a2]:
    label = "True Lag" if a==a1 else None
    a.set_yticklabels([])
    a.grid()
    a.legend()
    a.set_xlim(0,lagplot.max())
    a.axvline(mock.lag, c='k', ls='--', zorder=-1, label=label)

f.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("./smoothing.png", bbox_inches="tight")
plt.show()