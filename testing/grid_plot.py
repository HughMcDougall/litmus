'''
Attempt at demonstrating a litmus scan
'''
from litmus_rm import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.use('TkAgg')

key1, key2 = 'lag', 'logtau'
keys = [key1, key2]
N = 128

mock = litmus_rm.mocks.mock_C.copy(E=[.25, .25], cadence=[30, 30], maxtime=360 * 2, season=60)
mock.plot()

model = litmus_rm.models.GP_simple()
model.set_priors({'lag': [0, 300], 'logtau': [4, 10]})
model.set_priors(mock.params() | {key: model.prior_ranges[key] for key in keys})
data = model.lc_to_data(mock.lc_1, mock.lc_2)

print("Settup done")
Xl, Yl = [np.linspace(*model.prior_ranges[key], N) for key in keys]
X, Y = np.meshgrid(Xl, Yl)
params_grid = {key1: X.reshape(N * N), key2: Y.reshape(N * N)}
params = litmus_rm._utils.dict_extend(mock.params(), params_grid)

print("Doing evals")
Z = model.log_density(params, data)
Z = Z.reshape(N, N)

I, J = np.argmax(Z, axis=0), np.argmax(Z, axis=1)
Xbest, Ybest = X[0, :], Y[I, 0]
Zbest = Z[:, :].max(axis=0)
params_grid = {key1: Xbest, key2: Ybest}
params_best = params = litmus_rm._utils.dict_extend(mock.params(), params_grid)
Zcurve = model.log_density_hess(params_best, data)
Zcurve = np.array([c[1, 1] for c in Zcurve])
E = np.sqrt(-Zcurve) ** -1

select = (Zbest - Zbest.max() > -np.log(8)) * Zcurve < 0
select = np.argwhere(select).flatten()

# ---------------------------------
plt.rc('font', size=12)

print("Doing plot")
# c1, c2, c3 = 'plum', 'lightsalmon', 'skyblue'
c1, c2, c3, c4 = 'skyblue', 'lightsalmon', 'plum', 'w'
a, b = 4, 4
f1, f2, f3 = [plt.figure(figsize=(a, b)),
              plt.figure(figsize=(a, b)),
              plt.figure(figsize=(a, b))]
for j, f in enumerate([f1, f2, f3]):
    plt.figure(f)
    ax = plt.axes(projection='3d', computed_zorder=False)
    ax.view_init(elev=30, azim=-55, roll=0)

    ax.set_xlabel('$\Delta t$', fontsize=12)
    ax.set_ylabel('$\ln (\\tau)$', fontsize=12)
    ax.set_zlabel('Posterior Density', fontsize=12)
    ax.set_zticklabels([])

    for waxis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        waxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.grid(False)
    ax.set_box_aspect([1, 1, 0.25])

    # --------------------
    I_forplot = select[::12]
    if j != 2: ax.plot_wireframe(X, Y, np.exp(Z - Z.max()), color=c1, lw=1, alpha=0.75, rstride=6, cstride=4,
                                 zorder=-1)
    ax.plot_wireframe(X, Y, np.zeros_like(Z), color='w', lw=2, alpha=0.25, rstride=6, cstride=4, zorder=-10)
    if j == 0: continue
    ax.plot(Xbest, Ybest, np.exp(Zbest - Z.max()), c=c2, lw=2, zorder=np.inf)
    ax.plot(Xbest, Ybest, np.zeros_like(Xbest), c=c2, lw=2, zorder=-np.inf, alpha=0.25)
    ax.scatter(Xbest[I_forplot], Ybest[I_forplot], np.exp((Zbest[I_forplot] - Z.max())), c=c4, lw=2, zorder=np.inf, s=4)
    if j == 1: continue

    for i in I_forplot:
        x = np.ones(N) * Xbest[i]
        y = np.linspace(Ybest[i] - 3 * E[i], Ybest[i] + 3 * E[i], N)
        z = np.exp((Zbest[i] - Z.max()) - 1 / 2 * ((y - Ybest[i]) / E[i]) ** 2)

        Y_fill = np.tile(y, (N, 1))
        X_fill = np.ones([N, N]) * Xbest[i]
        Z_fill = np.tile(np.linspace(0, 1, N), (N, 1)).T
        for k in range(N):
            Z_fill[:, k] *= z[k]

        ax.plot(x, y, z, c=c3, lw=2, zorder=2 * i + 1)
        ax.plot(x, y, z * 0, c=c3, lw=1, zorder=2 * i + 1)
        ax.plot_surface(X_fill, Y_fill, Z_fill, color=c3, alpha=0.5, lw=0, zorder=2 * i)
        ax.plot([Xbest[i]] * 2, [Ybest[i]] * 2, [np.exp((Zbest[i] - Z.max())), 0], c=c4, lw=2, zorder=2 * i,
                alpha=0.75)
    ax.plot(Xbest, Ybest, np.zeros_like(Xbest), c=c2, lw=2, zorder=-np.inf, alpha=0.75)
for i, f in enumerate([f1, f2, f3]):
    f.tight_layout()
    f.savefig("Contour_fig_%i.png" % i, bbox_inches='tight', dpi=300)
plt.show()

# -----------------------------------------------
if False:
    print("Doing heatmap")

    plt.figure()
    plt.imshow(np.exp(Z - Z.max()), extent=np.array([model.prior_ranges[key] for key in keys]).flatten(),
               origin='lower',
               aspect='auto')
    ALPHA = np.exp(Zbest - Zbest.max()) * E
    ALPHA /= ALPHA.max()
    ALPHA = 0.5 + ALPHA * 0.5
    for i in select:
        plt.errorbar(Xbest[i], Ybest[i], E[i], c='r', capsize=2, fmt='none', alpha=ALPHA[i])
    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.show()

print("All groovy")
