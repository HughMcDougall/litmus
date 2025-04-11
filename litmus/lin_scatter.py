import numpy as np
from typing import Any
from nptyping import NDArray


def linscatter(X: np.array, Y: np.array, N: int) -> np.ndarray[Any, np.float64]:
    '''
    For some time series {X,Y}, returns N samples that are distributed along X with linear interpolation
    :param X: X values
    :param Y: Y values
    :param N: Number of samples
    :return: 1D numpy array of samples along X
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


def expscatter(X: np.array, Y: np.array, N) -> np.ndarray[Any, np.float64]:
    '''
    For some time series {X,Y}, returns N samples that are distributed along X with log-linear interpolation
    :param X: X values
    :param Y: Y values
    :param N: Number of samples
    :return: 1D numpy array of samples along X
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

