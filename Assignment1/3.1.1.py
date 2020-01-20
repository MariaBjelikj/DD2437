import numpy as np
import matplotlib.pyplot as plt
import time
from DataGeneration import linearly_separable_data

CONVERGENCE = 0.001
ITERATIONS = 50
SAMPLES = 100
SEPARATION = 0.5
ETA = 0.001
D_BATCH = 1
D_SEQUENTIAL = 2
PERCEPTRON = 3

def delta_rule_batch(x, t, w):
    w += - ETA * np.dot(np.dot(w, x) - t, np.transpose(x)) # delta rule
    return w

def delta_rule_sequential(x, t, w):
    delta_w = w[:]
    for i in range(x.shape[1]):
        delta_w = delta_w - ETA * np.dot(np.dot(w, x[:, i, None]) - t[:, i], np.transpose(x[:, i, None]))
        
    return delta_w
    
def perceptron_learning(x, t, w):
    # https://en.wikipedia.org/wiki/Perceptron
    for i in range(0, x.shape[1]):

        activation = w[0][len(x[:,i]) - 1]
        for j in range(0, len(x[:,i])):
            activation += w[0][j] * x[:,i][j] # activation is the sum of all the weights * nodes
            
        if activation >= 0:
            prediction = 1.0
        else :
            prediction = -1.0

        for k in range(0, len(x[:,i])):
            w[0][k] = w[0][k] + (ETA * (t[:,i] - prediction) / 2 * x[:,i][k]) # error calculated as (real target - prediction) / 2

    return w

def plot_boundary(x_grid, x, t, w, algorithm):

    title = ""
    for i in range(ITERATIONS):

        if (algorithm == D_BATCH):
            w = delta_rule_batch(x, t, w)
            title = "Delta Batch"
        elif (algorithm == D_SEQUENTIAL):
            w = delta_rule_sequential(x, t, w)
            title = "Delta Sequential"
        elif (algorithm == PERCEPTRON):
            w = perceptron_learning(x, t, w)
            title = "Simple Perceptron"
        else:
            print("Unknown algorithm. Exiting...")
            return

        plt.clf()
        axes = plt.gca()
        axes.set_xlim(min(x[0,:]) - 0.5, max(x[0,:]) + 0.5)
        axes.set_ylim(min(x[1,:]) - 0.5, max(x[1,:]) + 0.5)
        plt.plot(x_grid, calculateBoundary(x_grid, w), label="Boundary", color='red')
        plt.title(title + ": Iteration %i" %(i+1))
        plt.xlabel("$\mathregular{X_1}$ coordinate")
        plt.ylabel("$\mathregular{X_2}$ coordinate")
        plt.legend()
        plt.scatter(x[0,:], x[1,:], c=t[0,:])
        if (i == 0 or i == ITERATIONS - 1):
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.01)

def plot_data(X, t):
    plt.scatter(X[0,:], X[1,:], c=t[0,:])
    plt.show()
    
def calculateBoundary(x, w):
    return (-w[:,0] * x - w[:,2]) / w[:,1] # Wx = 0

mA = [ 1.0, 0.5]
mB = [-1.0, 0.0] 
sigmaA = 0.5
sigmaB = 0.5
x, t = linearly_separable_data(mA, sigmaA, mB, sigmaB)
x_grid = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / SAMPLES)
w = np.random.rand(t.shape[0], x.shape[0])
plot_boundary(x_grid, x, t, w, PERCEPTRON)