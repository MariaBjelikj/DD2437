import numpy as np
import Perceptron_algorithm as alg
import Part1 as data

n_hidden = 8
x, t, x_test, t_test = data.generate_data(f_type="sin2x", noise=True)
x = x.reshape(1, len(x))
t = t.reshape(1, len(t))
x_test = x_test.reshape(1, len(x_test))
t_test = t_test.reshape(1, len(t_test))
x = np.vstack((x, np.ones((x.shape[1]))))

# x_train, t_train, x_grid, y_grid, x_test, t_test = dg.gaussian_data(0.2)
v = np.random.normal(0, 1, (t.shape[0], n_hidden))
w = np.random.normal(0, 1, (n_hidden, x.shape[0]))
dv = np.zeros((t.shape[0], n_hidden))
dw = np.zeros((n_hidden, x.shape[0]))
x_grid = np.arange(min(x[0, :]), max(x[0, :]), (max(x[0, :]) - min(x[0, :])) / len(x[0]))
x_grid = x_grid.reshape(1, len(x_grid))
alg.function_approximation(x, w, v, dw, dv, t, x_grid, n_hidden, x_test, t_test)
