# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:08:18 2020

@author: bjeli
"""
import numpy as np
import matplotlib.pyplot as plt
import time

CONVERGENCE = 0.001
ITERATIONS = 50
SAMPLES = 100
SEPARATION = 0.33

def linearly_separable_data(mA, sigmaA, mB, sigmaB):
    """
    Generates linearly separable data for classification given the means mA
    and mB and variances sigmaA and sigmaB for two classes A and B.
    Parameters:
    mA - mean for class A
    sigmaA - variance for class A
    mB - mean for class B
    sigmaB - variance for class B
    """

    classA_x1 = np.random.randn(1,SAMPLES) * sigmaA + mA[0] + SEPARATION
    classA_x2 = np.random.randn(1,SAMPLES) * sigmaA + mA[1] + SEPARATION
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])
    
    
    classB_x1 = np.random.randn(1,SAMPLES) * sigmaB + mB[0] - SEPARATION
    classB_x2 = np.random.randn(1,SAMPLES) * sigmaB + mB[1] - SEPARATION
    classB = np.vstack([classB_x1, classB_x2])
    labelsB = -np.ones([1,SAMPLES])
    
    data = np.hstack([classA, classB])
    targets = np.hstack([labelsA, labelsB])
    
    X = np.zeros([3, 2 * SAMPLES])
    t = np.zeros([1, 2 * SAMPLES])
    shuffle = np.random.permutation(2 * SAMPLES)
    
    for i in shuffle:
        X[:2,i] = data[:2, shuffle[i]] # shuffle data
        X[2,i] = 1 # add bias
        t[0,i] = targets[0, shuffle[i]]
    
    return X, t


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

__x = np.arange(min(x[0,:]), max(x[0,:]), (max(x[0,:]) - min(x[0,:])) / SAMPLES)
eta = 0.001
w = np.random.rand(t.shape[0], x.shape[0])

for i in range(ITERATIONS):
    delta_w = -eta * np.dot(np.dot(w, x) - t, np.transpose(x))
    w += delta_w

    plt.clf()
    axes = plt.gca()
    axes.set_xlim(min(x[0,:]) - 0.5, max(x[0,:]) + 0.5)
    axes.set_ylim(min(x[1,:]) - 0.5, max(x[1,:]) + 0.5)
    plt.plot(__x, calculateBoundary(__x, w), label="Boundary", color='red')
    plt.title("Simple Perceptron: Iteration %i" %(i+1))
    plt.xlabel("X-Coord")
    plt.ylabel("Y-Coord")
    plt.legend()
    plt.scatter(x[0,:], x[1,:], c=t[0,:])
    if (i == 0 or i == ITERATIONS - 1):
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(0.05)