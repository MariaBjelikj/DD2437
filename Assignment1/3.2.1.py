import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

SAMPLES = 100

mean = np.array([0,1])
sigma = 1
n_hidden = 5
#x, t = dg.linearly_separable_data([1.0, 0.5], 0.5, [-1.0, 0], 0.5)
x, t = dg.new_data_generation([1.0, 0.3], [0, -0.1], 0.2, 0.3)
v = np.random.rand(t.shape[0], n_hidden)
w = np.random.rand(n_hidden, x.shape[0])
dv = np.random.rand(t.shape[0], n_hidden)
dw = np.random.rand(n_hidden, x.shape[0])
x_grid, y_grid = np.meshgrid(np.arange(min(x[0,:]), max(x[0,:]), 
(max(x[0,:]) - min(x[0,:])) / SAMPLES), np.arange(min(x[1,:]), max(x[1,:]), (max(x[1,:]) - min(x[1,:])) / SAMPLES))
alg.plot_boundary_multilayer(x, w, v, dw, dv, t, x_grid, y_grid)

