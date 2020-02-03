data = __import__('3')
import numpy as np
import matplotlib.pyplot as plt
import time
#import DataGeneration as dg
import Perceptron_algorithm as alg
import Constants as cte

n_hidden = 8
x, t, x_test, t_test = data.generate_data(f_type="sin2x", noise=True)
x = x.reshape(1,len(x))
t = t.reshape(1,len(t))

#x_train, t_train, x_grid, y_grid, x_test, t_test = dg.gaussian_data(0.2)
v = np.random.normal(0,1,(t.shape[0], n_hidden))
w = np.random.normal(0,1,(n_hidden, x.shape[0]))
dv = np.zeros((t.shape[0], n_hidden))
dw = np.zeros((n_hidden, x.shape[0]))

x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / len(x[0]))
x_grid = x_grid.reshape(1, len(x_grid)).T

alg.function_approximation(x, w, v, dw, dv, t, x_grid,  n_hidden, x_test, t_test)



