import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import linearly_separable_data
import Algorithms as alg
import Constants as cte


def main():
    mA = [ 1.0, 0.5]
    mB = [-1.0, 0.0] 
    sigmaA = 0.5
    sigmaB = 0.5
    x, t = linearly_separable_data(mA, sigmaA, mB, sigmaB)
    x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / cte.SAMPLES)
    w = np.random.rand(t.shape[0], x.shape[0])

    # Case: ∆ Batch
    alg.plot_boundary(x_grid, x, t, w, cte.D_BATCH)

    # Case: ∆ Sequential
    alg.plot_boundary(x_grid, x, t, w, cte.D_SEQUENTIAL)

    # Case: Simple Perceptron
    alg.plot_boundary(x_grid, x, t, w, cte.PERCEPTRON)

if __name__ == "__main__":
    main()