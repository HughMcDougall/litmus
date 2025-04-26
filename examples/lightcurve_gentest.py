import litmus
import numpy as np
import matplotlib.pyplot as plt
from litmus._utils import *

from litmus.mocks import mock_A, mock_B, mock_C

mock = litmus.mocks.mock(lag=180,
                         E=[0.1, 0.25],
                         tau=400)
mock.plot()

lc_1, lc_2 = mock.lc_1, mock.lc_2

model = litmus.models.GP_simple(verbose=True,
                                prior_ranges=mock.params() | {'lag': [0, 800], 'logtau': [np.log(10), np.log(5000)]})
data = model.lc_to_data(lc_1, lc_2)
Tpred = np.linspace(-1000, 2500, 512)
# --------------------------------------

p = mock.params()
pred1_true, pred2_true = model.make_lightcurves(data, params=p, Tpred=Tpred, num_samples=1)

def make_plot(pred1, pred2):
    print("Predictions Clear. Plotting")
    c0 = 'midnightblue'
    c1, c2 = 'navy', 'orchid'

    f, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    lc_1.plot(axis=a1, show=False, c=c1, capsize=2, label="Measurements")
    lc_2.plot(axis=a2, show=False, c=c2, capsize=2, label="Measurements")

    a1.plot(mock.lc.T, mock.lc.Y, c=c0, lw=0.5, zorder=-6, label='True Signal')
    a2.plot(mock.lc.T + mock.lag, mock.lc.Y, c=c0, lw=0.5, zorder=-6)

    a1.fill_between(pred1.T, pred1.Y - pred1.E, pred1.Y + pred1.E, alpha=0.25, color=c1,
                    label="Constrained Lightcurve, Continuum")
    a2.fill_between(pred2.T, pred2.Y - pred2.E, pred2.Y + pred2.E, alpha=0.25, color=c2,
                    label="Constrained Lightcurve, Response")
    a1.fill_between(pred1.T, pred1.Y - 2 * pred1.E, pred1.Y + 2 * pred1.E, alpha=0.125, color=c1)
    a2.fill_between(pred2.T, pred2.Y - 2 * pred2.E, pred2.Y + 2 * pred2.E, alpha=0.125, color=c2)

    r = 0.1
    a1.fill_between(pred1.T, pred1.Y - 2 * pred1.E - r, pred1.Y + 2 * pred1.E + r, alpha=1.0, color='w', zorder=-9)
    a2.fill_between(pred2.T, pred2.Y - 2 * pred2.E - r, pred2.Y + 2 * pred2.E + r, alpha=1.0, color='w', zorder=-9)

    f.supxlabel("Time (Days)")
    f.supylabel("Signal (Arb Units)")

    for a in a1, a2:
        a.grid()
        a.set_yticklabels([])
        a.set_ylim(-3, 3)

        if mock.season != 0:
            tmax = Tpred.max()
            nyears = int(tmax // (mock.season * 2) + 1)
            for i in range(nyears):
                a.axvspan((i + 1 / 2 - 1 / 2) * mock.season * 2, (i + 1 - 1 / 2) * mock.season * 2,
                          ymin=0, ymax=1, alpha=0.125, color='royalblue',
                          zorder=-10,
                          label=None)
            a.legend()

    a1.set_xlim(0, 2000)
    f.tight_layout()

    print("Plots done")

    return (f)

f = make_plot(pred1_true, pred2_true)
f.suptitle("Constrained Lightcurves at true values")
f.tight_layout()
f.show()

my_fitter = litmus.fitting_methods.SVI_scan(model, verbose=True, debug=False)
my_fitter.fit(lc_1, lc_2)
p_varied = my_fitter.get_samples()

lt = litmus.LITMUS(my_fitter)
lt.lag_plot()
lt.plot_parameters()
plt.show()