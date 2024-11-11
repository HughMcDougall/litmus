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
    R = np.random.rand(N)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reverse = False
    X = [9, 10, 10.5]
    Y = [0.2, 1.5, 1.5]

    if reverse:
        X = np.array(X)[::-1]
        Y = np.array(Y)[::-1]

    Y = np.array(Y) / np.trapz(Y, X)

    plt.plot(X, Y)
    samples = linscatter(X, Y, 50_000) + X[1]
    plt.hist(samples, bins=256, density=True)
    plt.show()
