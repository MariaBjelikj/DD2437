import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

n_hidden = 25
x, t, x_grid, y_grid = dg.gaussian_data(0.5)
v = np.random.normal(0,1,(t.shape[0], n_hidden))
w = np.random.normal(0,1,(n_hidden, x.shape[0]))
dv = np.zeros((t.shape[0], n_hidden))
dw = np.zeros((n_hidden, x.shape[0]))
alg.function_approximation(x, w, v, dw, dv, t, x_grid, y_grid, n_hidden)