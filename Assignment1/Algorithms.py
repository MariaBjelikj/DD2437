import numpy as np
import matplotlib.pyplot as plt

CONVERGENCE = 0.001
ITERATIONS = 20
ETA = 0.001
D_BATCH = 1
D_SEQUENTIAL = 2
PERCEPTRON = 3

def delta_rule_batch(x, t, w):
    w_aux = w.copy()
    w_aux += - ETA * np.dot(np.dot(w_aux, x) - t, np.transpose(x)) # delta rule

    return w_aux

def delta_rule_sequential(x, t, w):
    delta_w = w.copy()
    for i in range(x.shape[1]):
        delta_w = delta_w - ETA * np.dot(np.dot(w, x[:, i, None]) - t[:, i], np.transpose(x[:, i, None]))
        
    return delta_w
    
def perceptron_learning(x, t, w):
    # https://en.wikipedia.org/wiki/Perceptron
    w_aux = w.copy()
    for i in range(0, x.shape[1]):
        activation = w_aux[0][len(x[:,i]) - 1]
        for j in range(0, len(x[:,i])):
            activation += w_aux[0][j] * x[:,i][j] # activation is the sum of all the weights * nodes
            
        if (activation >= 0):
            prediction = 1.0
        else:
            prediction = -1.0

        for k in range(0, len(x[:,i])):
            w_aux[0][k] = w_aux[0][k] + (ETA * (t[:,i] - prediction) / 2 * x[:,i][k]) # error calculated as (real target - prediction) / 2

    return w_aux

def calculateBoundary(x, w):
    w_aux = w.copy()
    return (-w_aux[:,0] * x - w_aux[:,2]) / w_aux[:,1] # Wx = 0

def plot_boundary(x_grid, x, t, w, algorithm):
    w_aux = w.copy()
    title = ""
    for i in range(ITERATIONS):
        if (algorithm == D_BATCH):
            w_aux = delta_rule_batch(x, t, w_aux)
            title = "Delta Batch"
        elif (algorithm == D_SEQUENTIAL):
            w_aux = delta_rule_sequential(x, t, w_aux)
            title = "Delta Sequential"
        elif (algorithm == PERCEPTRON):
            w_aux = perceptron_learning(x, t, w_aux)
            title = "Simple Perceptron"
        else:
            print("Unknown algorithm. Exiting...")
            return

        plt.clf()
        axes = plt.gca()
        axes.set_xlim(min(x[0,:]), max(x[0,:]))
        axes.set_ylim(min(x[1,:]), max(x[1,:]))
        plt.plot(x_grid, calculateBoundary(x_grid, w_aux), label="Boundary", color='red')
        plt.title(title + ": Iteration %i" %(i+1))
        plt.xlabel("$\mathregular{X_1}$ coordinate")
        plt.ylabel("$\mathregular{X_2}$ coordinate")
        plt.legend()
        plt.scatter(x[0,:], x[1,:], c=t[0,:])
        if (i == 0 or i == ITERATIONS - 1):
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.005)

def plot_data(x, t):
    plt.scatter(x[0,:], x[1,:], c=t[0,:])
    plt.show()

def run_algorithms(x_grid, x, t, w):
    # Case: ∆ Batch
    plot_boundary(x_grid, x, t, w, D_BATCH)

    # Case: ∆ Sequential
    plot_boundary(x_grid, x, t, w, D_SEQUENTIAL)

    # Case: Simple Perceptron
    plot_boundary(x_grid, x, t, w, PERCEPTRON)