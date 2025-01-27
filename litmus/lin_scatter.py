import numpy as np


def linscatter(X: np.array, Y: np.array, N):
    '''
    :param X:
    :param Y:
    :return Xshift:
    '''

    dx = np.array([X[0] - X[1], X[2] - X[1]])
    dy = np.array([Y[0] - Y[1], Y[2] - Y[1]])
    dx, dy = dx[dx.argsort()], dy[dx.argsort()]

    weight_leftright = abs((Y[1] + dy / 2.0) * dx)
    weight_leftright /= weight_leftright.sum()

    leftright = np.random.choice([0, 1], replace=True, size=N, p=weight_leftright)

    DX, DY = dx[leftright], dy[leftright]
    YBAR = Y[1] + DY / 2
    c1, c2 = YBAR / DY, Y[1] / DY

    CDF = np.random.rand(N)

    # todo - this throws an error because is evaluates the first branch in full. Find a way to suppress.
    Xshift = np.where(DY != 0,
                      np.sign(DY) * np.sqrt(CDF * c1 * 2 + (c2) ** 2) - c2,
                      CDF
                      )
    Xshift = Xshift * DX

    return Xshift


def expscatter(X: np.array, Y: np.array, N):
    '''
    :param X:
    :param Y:
    :return Xshift:
    '''

    dx = np.array([X[0] - X[1], X[2] - X[1]])
    dy = np.array([Y[0] - Y[1], Y[2] - Y[1]])
    dE = np.log(np.array([Y[0] / Y[1], Y[2] / Y[1]]))
    dx, dy, dE = dx[dx.argsort()], dy[dx.argsort()], dE[dx.argsort()]

    weight_leftright = abs(dx * dy / dE)

    if dy[0] == 0: weight_leftright[0] = abs(dx[0] * Y[1])
    if dy[1] == 0: weight_leftright[1] = abs(dx[1] * Y[1])


    weight_leftright /= weight_leftright.sum()


    leftright = np.random.choice([0, 1], replace=True, size=N, p=weight_leftright)

    DX, DY, DE = dx[leftright], dy[leftright], dE[leftright]

    CDF = np.random.rand(N)

    # todo - this throws an error because is evaluates the first branch in full. Find a way to suppress.
    Xshift = np.where(DY != 0,
                      np.log(CDF * DY / Y[1] + 1) / DE,
                      CDF
                      )

    Xshift = Xshift * DX

    return Xshift


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = [0, 10, 18]
    Y = [10, 2.1, 0.01]


    samples_lin = linscatter(X, Y, 50_000) + X[1]
    samples_exp = expscatter(X, Y, 50_000) + X[1]

    Y = np.array(Y) / np.trapz(Y, X)

    Xinterp = np.linspace(X[0], X[2], 1024)
    Y2 = np.exp(np.interp(Xinterp, X, np.log(Y)))
    Y2 /= np.trapz(Y2, Xinterp)

    f, (a1, a2) = plt.subplots(1, 2, sharey=True, sharex=True)
    a1.plot(X, Y), a2.plot(Xinterp, Y2)
    a1.hist(samples_lin, bins=256, density=True, alpha=0.5)
    a2.hist(samples_exp, bins=256, density=True, alpha=0.5)

    a1.set_title("Linear Scatter")
    a2.set_title("Exp Scatter")
    a1.grid(), a2.grid()

    a1.set_xlim(X[0], X[-1])
    plt.show()
