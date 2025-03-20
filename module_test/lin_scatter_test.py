if __name__ == "__main__":
    from litmus.lin_scatter import *
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
