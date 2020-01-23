import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess as sp

CONVERGENCE = 0.1
ITERATIONS = 20
ITERATIONS_MULTI = 10000
ETA = 0.1
D_BATCH = 1
D_SEQUENTIAL = 2
PERCEPTRON = 3
ALPHA = 0.9

def delta_rule_batch(x, t, w):
    w_aux = w.copy()
    w_aux += - ETA * np.dot(np.dot(w_aux, x) - t, np.transpose(x)) # delta rule

    return w_aux

def delta_rule_sequential(x, t, w):
    delta_w = w.copy()
    for i in range(x.shape[1]):
        delta_w = delta_w - ETA * np.dot(np.dot(delta_w, x[:, i, None]) - t[:, i], np.transpose(x[:, i, None]))
        
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

# ---------------------------- MULTILAYER PERCEPTRON ---------------------------- #

def phi(x):
    phi = 2 / (1 + np.exp(-x)) - 1
    return phi

def forward_pass(x, w, v):
    h_in = np.dot(w, x)
    phi_h = phi(h_in)
    h_out =  np.vstack((phi_h))
    o_in = np.dot(v, h_out)
    o_out = phi(o_in)
    
    return h_out, o_out
    
def backward_pass(t, w, v, o, h):    
    delta_o = (o - t) * ((1 + o) * (1 - o)) * 0.5
    delta_h = np.dot(np.transpose(v), delta_o) * ((1 + h) * (1 - h)) * 0.5
    delta_h = delta_h[range(delta_h.shape[0]),:]    #remove extra rows
    
    return delta_o, delta_h

def weight_update(x, dw, dv, delta_h, delta_o, h, w, v):
    dw = (dw * ALPHA) - np.dot(delta_h, np.transpose(x)) * (1 - ALPHA)
    dv = (dv * ALPHA) - np.dot(delta_o, np.transpose(h)) * (1 - ALPHA)
    W = w + dw * ETA
    V = v + dv * ETA
    return W, V

def compute_mse(o, t):
    return np.sum(np.square(o - t)) / len(o)

def compute_misclassifications(o, t):
    cnt = 0
    for i in range(o.shape[1]):
        if ((o[0,i] > 0 and t[0,i] < 0) or (o[0,i] < 0 and t[0,i] > 0)):
            cnt += 1
    return cnt, cnt / o.shape[1]

def plot_boundary_multilayer(x, w, v, dw, dv, t, x_grid, y_grid, x_test, t_test):
    z = 0
    cnt = 0
    error = float('inf')
    pbar = tqdm(total=100)
    learning_training = list()
    learning_testing = list()
    while (cnt < ITERATIONS_MULTI and error > CONVERGENCE):
        pbar.update(round(1/ITERATIONS_MULTI * 100, 2))
        h, o = forward_pass(x, w, v)
        delta_o, delta_h = backward_pass(t, w, v, o, h)
        w, v = weight_update(x, dw, dv, delta_h, delta_o, h, w, v)
        x_input = np.c_[x_grid.ravel(), y_grid.ravel()]
        x_input = np.transpose(x_input)
        x_input = np.vstack((x_input, np.ones((x_input.shape[1]))))
        error = compute_mse(forward_pass(x, w, v)[1], t)
        learning_training.append(error)
        learning_testing.append(compute_mse(forward_pass(x_test, w, v)[1], t_test))
        z = forward_pass(x_input, w, v)[1]
        z = z.reshape(x_grid.shape)
        cnt += 1
    pbar.close()
    sp.call('clear',shell=True)
    plt.clf()
    axes = plt.gca()
    axes.set_xlim(min(x_test[0,:]), max(x_test[0,:]))
    axes.set_ylim(min(x_test[1,:]), max(x_test[1,:]))
    plt.contour(x_grid, y_grid, z, levels=[0])
    if (cnt < ITERATIONS_MULTI):
        plt.title("Multilayer Perceptron: Convergence found in %i iterations" %cnt)
    else:
        plt.title("Multilayer Perceptron: %i iterations" %cnt)
    plt.xlabel("$\mathregular{X_1}$ coordinate")
    plt.ylabel("$\mathregular{X_2}$ coordinate")
    plt.scatter(x_test[0,:], x_test[1,:], c=t_test[0,:])
    plt.show()
    np.set_printoptions(precision=6)
    print("Final error: %f" %compute_mse(forward_pass(x_test, w, v)[1], t_test))
    data, ratio = compute_misclassifications(forward_pass(x_test, w, v)[1], t_test)
    print("Misclassified data: %i" %data)
    print("Misclassified ratio: %f" %ratio)
    plt.plot(range(cnt), learning_training, color="blue", label="Error training")
    plt.plot(range(cnt), learning_testing, color="red", label="Error testing")
    plt.xlabel("Epochs")
    plt.ylabel("Error (MSE)")
    plt.title("Learning curves: Training vs Testing")
    plt.legend()
    plt.show()
