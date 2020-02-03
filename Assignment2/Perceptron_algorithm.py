import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import Constants as cte


def phi(x):
    phi_val = 2 / (1 + np.exp(-x)) - 1
    return phi_val


def forward_pass(x, w, v):
    h_in = np.dot(w, x)
    phi_h = phi(h_in)
    h_out = np.vstack(phi_h)
    o_in = np.dot(v, h_out)
    o_out = phi(o_in)

    return h_out, o_out


def backward_pass(t, v, o, h):
    delta_o = (o - t) * ((1 + o) * (1 - o)) * 0.5
    delta_h = np.dot(np.transpose(v), delta_o) * ((1 + h) * (1 - h)) * 0.5
    delta_h = delta_h[range(delta_h.shape[0]), :]

    return delta_o, delta_h


def weight_update(x, dw, dv, delta_h, delta_o, h, w, v):
    dw = (dw * cte.ALPHA) - np.dot(delta_h, np.transpose(x)) * (1 - cte.ALPHA)
    dv = (dv * cte.ALPHA) - np.dot(delta_o, np.transpose(h)) * (1 - cte.ALPHA)
    new_w = w + dw * cte.ETA
    new_v = v + dv * cte.ETA
    return new_w, new_v


def compute_mse(o, t):
    return np.sum(np.square(o - t)) / len(o)


def function_approximation(x_train, w, v, dw, dv, t_train, x_grid, n_hidden, x_test, t_test):
    cnt = 0
    z = 0
    error = float('inf')
    pbar = tqdm(total=100)
    while cnt < cte.ITERATIONS_MULTI and error > cte.CONVERGENCE:
        pbar.update(round(1 / cte.ITERATIONS_MULTI * 100, 2))
        h, o = forward_pass(x_train, w, v)
        delta_o, delta_h = backward_pass(t_train, v, o, h)
        w, v = weight_update(x_train, dw, dv, delta_h, delta_o, h, w, v)
        x_input = x_grid
        x_input = np.vstack((x_input, np.ones((x_input.shape[1]))))
        error = compute_mse(forward_pass(x_train, w, v)[1], t_train)
        z = forward_pass(x_input, w, v)[1]
        z = z.reshape(x_grid.shape)
        cnt += 1
    pbar.close()
    ax = plt.gca()
    plt.plot(x_grid.T, z.T, label='Predicted')  # Prediction
    plt.plot(x_test.T, t_test.T, label='Real')  # Original data
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title("Prediction with %i nodes" % n_hidden)
    plt.legend()
    plt.show()
