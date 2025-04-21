
# ================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from litmus.mocks import *

    print("Running mock test code")

    #-------------
    mock_A_01, mock_A_02, lag_A = mock_A.lc_1, mock_A.lc_2, mock_A.lag
    mock_B_00 = mock_B.lc
    mock_B_01 = mock_B.lc_1
    mock_B_02 = mock_B.lc_2
    lag_B = mock_B.lag

    mock_C_00 = mock_C.lc
    mock_C_01 = mock_C.lc_1
    mock_C_02 = mock_C.lc_2
    lag_C = mock_C.lag
    #-------------

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
