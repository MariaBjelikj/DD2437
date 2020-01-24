import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

n_hidden = 25
x_train, t_train, x_grid, y_grid, x_test, t_test = dg.gaussian_data(0.2)
v = np.random.normal(0,1,(t_train.shape[0], n_hidden))
w = np.random.normal(0,1,(n_hidden, x_train.shape[0]))
dv = np.zeros((t_train.shape[0], n_hidden))
dw = np.zeros((n_hidden, x_train.shape[0]))
alg.function_approximation(x_train, w, v, dw, dv, t_train, x_grid, y_grid, n_hidden, x_test, t_test)