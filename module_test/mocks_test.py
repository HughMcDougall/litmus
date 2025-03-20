
# ================================================
if __name__ == "__main__":
    print("Running mock test code")
    import matplotlib.pyplot as plt
    from litmus.mocks import *
    for x in mock_A, mock_B, mock_C:
        plt.figure()
        plt.title("Seasonal GP, lag = %.2f" % x.lag)

        x.plot(axis=plt.gca())

        plt.legend()
        plt.xlabel("Time (Days)")
        plt.ylabel("Signal Strength")
        plt.axhline(0.0, ls='--', c='k', zorder=-10)
        plt.axhline(1.0, ls=':', c='k', zorder=-10)
        plt.axhline(-1.0, ls=':', c='k', zorder=-10)

        plt.grid()
        plt.show()

        # ------------------
        plt.figure()
        x.corrected_plot(x.params(), axis=plt.gca(), true_args={'alpha': [0.15, 0.0]})
        plt.title("Corrected_plot for lag = %.2f" % x.lag)
        plt.grid()

        plt.show()
