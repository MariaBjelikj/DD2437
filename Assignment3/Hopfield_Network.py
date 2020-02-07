import numpy as np
import matplotlib.pyplot as plt

ITERATIONS = 100  # number of iterations for syncronious update


def generate_data(d_type):
    # For task 3.1

    if d_type == "original":
        x1d = [-1, -1, 1, -1, 1, -1, -1, 1]
        x2d = [-1, -1, -1, -1, -1, 1, -1, -1]
        x3d = [-1, 1, 1, -1, -1, 1, -1, 1]
        return np.vstack([x1d, x2d, x3d])

    elif d_type == "distorted":
        x1d = [1, -1, 1, -1, 1, -1, -1, 1]  # one bit error
        x2d = [1, 1, -1, -1, -1, 1, -1, -1]  # two bit error
        x3d = [1, 1, 1, -1, 1, 1, -1, 1]  # two bit error
        return np.vstack([x1d, x2d, x3d])

    elif d_type == "super_distorted":
        # more then half the data is distroted
        x1d = [1, 1, -1, -1, -1, 1, -1, 1]
        x2d = [1, 1, 1, -1, -1, -1, 1, -1]
        x3d = [1, -1, -1, 1, -1, 1, -1, -1]
        return np.vstack([x1d, x2d, x3d])


def weights(x):
    # Update weights for Little Model
    n = x.shape[0]  # number of patterns
    m = x.shape[1]  # number of neurons

    w = np.zeros([m, m])
    for i in range(n):
        # calculate weights
        x_i = x[i, :]
        w += np.outer(x_i.T, x_i)

    for i in range(m):
        w[i, i] = 0  # fill diagonal with 0

    # w = w / M # normalize, only if bias is used
    return w


def set_sign(x, w, index):
    x_sign = np.copy(x)
    for i in range(len(x[index, :])):
        aux = 0
        for j in range(len(x[index, :])):
            aux += w[i, j] * x_sign[index, j]
        if aux >= 0:
            x_sign[index, i] = 1
        else:
            x_sign[index, i] = -1
    return x_sign[index, :]


def update_syncroniously(x, w):
    # Update the weights synchroniously
    # aka "Little Model"
    x_current = np.copy(x)
    x_new = np.copy(x)

    # Iterate for convergence
    for iteration in range(ITERATIONS):
        for i in range(x.shape[0]):
            x_new[i, :] = set_sign(x_current, w, i)

        if np.all(x_new == x_current):  # check recall
            print("The network converged after {} iterations.".format(iteration))
            break  # the state is stable (convergence, break loop)

        x_current = np.copy(x_new)

    return x_current


def update_asyncroniously(x, w):
    # Update the weights synchroniously
    # aka "Little Model"
    x_current = np.copy(x)
    x_new = np.copy(x)

    # Iterate for convergence
    for iteration in range(ITERATIONS):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                idx = np.random.randint(0, x.shape[1])
                x_new[i, idx] = np.sign(x_current[i, :] @ w[idx])

        if np.all(x_new == x_current.T):  # check recall
            print("The network converged after {} iterations.".format(iteration))
            break  # the state is stable (convergence, break loop)

        x_current = x_new

    return x_current


def display(image):
    # For task 3.2

    # display images in shape (32, 32)
    plt.imshow(image.reshape(32, 32), interpolation="nearest")
    plt.show()