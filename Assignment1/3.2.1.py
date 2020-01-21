import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

SAMPLES = 100

mean = np.array([0,1])
sigma = 1
n_hidden = 5
x, t = dg.non_linearly_separable_data(mean, sigma)
v = np.random.rand(t.shape[0], n_hidden)
w = np.random.rand(n_hidden, x.shape[0])
dv = np.random.rand(t.shape[0], n_hidden)
dw = np.random.rand(n_hidden, x.shape[0])
x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / SAMPLES)
alg.plot_boundary_multilayer(x, w, v, dw, dv, t, x_grid)

