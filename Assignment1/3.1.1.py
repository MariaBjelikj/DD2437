# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:08:18 2020

@author: bjeli
"""
import numpy as np
import matplotlib.pyplot as plt

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
    n = 100
    
    classA_x1 = np.random.randn(1,n) * sigmaA + mA[0]
    classA_x2 = np.random.randn(1,n) * sigmaA + mA[1]
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,n])
    
    
    classB_x1 = np.random.randn(1,n) * sigmaB + mB[0]
    classB_x2 = np.random.randn(1,n) * sigmaB + mB[1]
    classB = np.vstack([classB_x1, classB_x2])
    labelsB = -np.ones([1,n])
    
    data = np.hstack([classA, classB])
    targets = np.hstack([labelsA, labelsB])
    
    X = np.zeros([3, 2 * n])
    t = np.zeros([1, 2 * n])
    shuffle = np.random.permutation(2 * n)
    
    for i in shuffle:
        X[:2,i] = data[:2, shuffle[i]] # shuffle data
        X[2,i] = 1 # add bias
        t[0,i] = targets[0, shuffle[i]]
    
    return X, t


def plot_data(X, t):
    plt.scatter(X[0,:], X[1,:], c=t[0,:])
    plt.show()




mA = [ 1.0, 0.5]
sigmaA = 0.5
mB = [-1.0, 0.0] 
sigmaB = 0.5
X, t = linearly_separable_data(mA, sigmaA, mB, sigmaB)

plot_data(X, t)