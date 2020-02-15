import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from tqdm import tqdm

ITERATIONS = 1000  # number of iterations for syncronious update


def iterative_patterns_accuracy(patterns, x_current):
    acc_counter = 0
    for j in range(x_current.shape[0]):
        aux = 0
        for i in range(patterns.shape[0]):
            if np.all(x_current[j] == patterns[i]):
                aux += 1
        if aux > 0:
            acc_counter += 1

    return acc_counter / patterns.shape[0]


def noised_images(percentages, data, pattern_position, counter, w, noised_iterations=1, return_data=False,iterative_patterns=False, theta=0):
    x_current = 0
    for i, perc in enumerate(percentages):
        for _ in range(noised_iterations):
            noised_data = np.copy(data)
            indexes = shuffle(np.arange(0, noised_data.shape[1]))
            for k in range(int(perc * noised_data.shape[1])):
                noised_data[pattern_position, indexes[k]] = -noised_data[pattern_position, indexes[k]]
            x_current = recall(noised_data[pattern_position:(pattern_position + 1), :], w, update_type="synchronous", convergence_type='energy', theta = theta)
            
            if iterative_patterns:
                counter = iterative_patterns_accuracy(noised_data[:pattern_position+1, :], x_current)
            
            else: 
                if np.all(x_current == data[pattern_position]):
                    counter[i] += 1
            
    if return_data:
        return counter, x_current

    return counter


def weights(x, weights_type=False, symmetrical=False, diagonal='',sparse_pattern = False):
    # Update weights for Little Model
    n = x.shape[0]  # number of patterns
    m = x.shape[1]  # number of neurons
    w = np.zeros([m, m])
    average_activity = 1/(n*m)* sum(sum(x))
    for i in range(n):
        # calculate weights
        if sparse_pattern:
            x_i = (x[i, :] - average_activity)
        else:
            x_i = x[i, :]
        w += np.outer(x_i.T, x_i)

    if diagonal == 'diagonal_0':
        np.fill_diagonal(w, 0)
                
    if weights_type == "normal":
        for i in range(m):
            for j in range(m):
                w[i, j] = random.normalvariate(0, 5)
    if symmetrical:
        w = 0.5 * (w + w.T)           

    # w = w / M # normalize, only if bias is used
    return w


def energy(state, w):
    return np.sum(-np.dot(state, np.dot(w, state.T)))


def display(image, title="", save=False, filename=''):
    # For task 3.2
    # display images in shape (32, 32) and rotate them so face is up
    plt.figure()
    plt.imshow(np.rot90(image.reshape(32, 32)), origin='lower', interpolation="nearest")
    if title != "":
        plt.title(title)
    if save:
        plt.imsave(filename, (np.rot90(image.reshape(32, 32))))
    plt.show()


def check_convergence_energy(x_new, w, energy_old, convergence_count):
    energy_new = energy(x_new, w)
    if energy_old == energy_new:  # If the energy hasn't changed, converged
        convergence_count += 1
    else:
        convergence_count = 0
    energy_old = np.copy(energy_new)
    return energy_old, convergence_count


def recall(x, w, update_type="synchronous", convergence_type="", asyn_type=False, sparse_pattern = False, theta = 0):
    """ 
    PARAMETERS:
    # update_type: can be "synchronous" or "asynchronous"
    # convergence_type: choose "energy" for task 3.3 and on
    # asyn_type: type of asynchronous update, "random" or sequential by default
    """

    x_current = 0
    if update_type == "synchronous":
        # Update the weights synchronously, aka "Little Model"
        x_current = np.copy(x)
        x_new = np.copy(x)
        # Compute energy of the initial state
        energy_old = energy(x, w)
        convergence_count = 0

        # Iterate for convergence
        for iteration in range(ITERATIONS):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if sparse_pattern:
                        update_rule = (x_current[i, :] @ w[j] - theta)
                        x_new[i, j] = 0.5 + 0.5*np.where((update_rule) >= 0, 1, -1)
                    else: 
                        x_new[i, j] = np.where((x_current[i, :] @ w[j]) >= 0, 1, -1)
                # Compute energy for this state

            if convergence_type == "energy":
                energy_old, convergence_count = check_convergence_energy(x_new, w, energy_old, convergence_count)
                if convergence_count > 3:
                    # print("The network converged after {} iterations.".format(iteration))
                    break
            else:
                if np.all(x_new == x_current):  # check recall
                    # print("The network converged after {} iterations.".format(iteration))
                    break
            x_current = np.copy(x_new)

    if update_type == "asynchronous":
        # Update the weights asynchroniously
        x_current = np.copy(x)
        x_new = np.copy(x)
        # Compute energy of the initial state
        energy_old = energy(x, w)
        convergence_count = 0

        # Iterate for convergence
        for iteration in range(ITERATIONS):
            for i in range(x.shape[0]):
                if asyn_type == "random":
                    idx = np.random.randint(0, x.shape[1])
                    x_new[i, idx] = np.where((x_current[i, :] @ w[idx]) >= 0, 1, -1)
                else:
                    for j in range(x.shape[1]):
                        x_new[i, j] = np.where((x_current[i, :] @ w[j]) >= 0, 1, -1)

            if convergence_type == "energy":
                energy_old, convergence_count = check_convergence_energy(x_new, w, energy_old, convergence_count)
                if convergence_count > 5:
                    print("The network converged after {} iterations.".format(iteration))
                    break
            else:
                if np.all(x_new == x_current):  # check recall
                    print("The network converged after {} iterations.".format(iteration))
                    break
            x_current = np.copy(x_new)

            # Task 3.2, plot every 100th iteration or so
            # to use this, comment out the parts for convergence so the network goes through all the iterations
            # iters = np.arange(0, 1000, 200)
            # if iteration in iters:
            # display(x_current, "Recall after {} iterations.".format(iteration))

    return x_current


def find_attractors(data, weight, update_type):
    data_updated = recall(data, weight, update_type=update_type)
    attractors = np.unique(data_updated, axis=0)
    return attractors

# TODO: Add bias
# Asynchronous update: Maybe we should change the order of how we update?
