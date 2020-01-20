import numpy as np
import matplotlib.pyplot as plt
import time

CONVERGENCE = 0.001
ITERATIONS = 50
SAMPLES = 100

def non_linearly_separable_data(m, sigma):
    """
    Generates non-linearly separable data for classification given the mean m
    and variance sigma for two classes A and B.
    Parameters:
    m - mean for both classes
    sigma - variance for both classes
    """

    classA_x1 = np.random.randn(1,SAMPLES) * sigma + m[0]
    classA_x2 = np.random.randn(1,SAMPLES) * sigma + m[1]
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])
    
    
    classB_x1 = np.random.randn(1,SAMPLES) * sigma + m[0]
    classB_x2 = np.random.randn(1,SAMPLES) * sigma + m[1]
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

def new_data_generation(m_a, m_b, sigma_a, sigma_b):
    classA_x1 = [np.random.randn(1, round(0.5 * SAMPLES)) * sigma_a - m_a[0],
    np.random.randn(1, round(0.5 * SAMPLES)) * sigma_a + m_a[0]]
    classA_x2 = np.random.randn(1,SAMPLES) * sigma_a + m_a[1]
    classA = np.vstack([classA_x1, classA_x2])
    labelsA = np.ones([1,SAMPLES])
    
    
    classB_x1 = np.random.randn(1,SAMPLES) * sigma_b + m_b[0]
    classB_x2 = np.random.randn(1,SAMPLES) * sigma_b + m_b[1]
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

m = [1.0, 0.5]
sigma = 0.5
x, t = non_linearly_separable_data(m, sigma)

# -------------------- FIRST PART -> Linear Separation for non-linearly separable data -------------------- #

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

# -------------------- SECOND PART -> Separation of non-linearly separable data -------------------- #

m_a = [1.0, 0.3]
m_b = [0.0, -0.1]
sigma_a = 0.2
sigma_b = 0.3
x, t = new_data_generation(m_a, m_b, sigma_a, sigma_b)

"""
Data removal conditions:
    - random 25% from each class
    - random 50% from classA
    - random 50% from classB
    - 20% from a subset of classA for which classA(1,:)<0 
        and 80% from a subset of classA for which classA(1,:)>0
"""
"""
# Class B
removal_pos = list()
while (len(removal_pos) < SAMPLES/2):
    aux = np.random.randint(low=0, high=2*SAMPLES-1)
    if (t[0,aux] == -1):
        if (aux not in removal_pos):
            removal_pos.append(aux)
print(removal_pos)
"""
