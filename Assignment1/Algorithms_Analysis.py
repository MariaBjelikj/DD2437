import numpy as np
import matplotlib.pyplot as plt
import Constants as cte

def delta_rule_batch(x, t, w):
    w_aux = w.copy()
    w_aux += - cte.ETA * np.dot(np.dot(w_aux, x) - t, np.transpose(x)) # delta rule

    return w_aux

def delta_rule_sequential(x, t, w):
    delta_w = w.copy()
    for i in range(x.shape[1]):
        delta_w = delta_w - cte.ETA * np.dot(np.dot(delta_w, x[:, i, None]) - t[:, i], np.transpose(x[:, i, None]))
        
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
            w_aux[0][k] = w_aux[0][k] + (cte.ETA * (t[:,i] - prediction) / 2 * x[:,i][k]) # error calculated as (real target - prediction) / 2

    return w_aux

def calculateBoundary(x, w):
    w_aux = w.copy()
    return (-w_aux[:,0] * x - w_aux[:,2]) / w_aux[:,1] # Wx = 0

def plot_boundary(x_grid, x, t, w, algorithm):
    w_aux = w.copy()
    title = ""
    accuracy = []
    
    for i in range(cte.ITERATIONS):
        if (algorithm == cte.D_BATCH):
            w_aux = delta_rule_batch(x, t, w_aux)
            title = "Delta Batch"
        elif (algorithm == cte.D_SEQUENTIAL):
            w_aux = delta_rule_sequential(x, t, w_aux)
            title = "Delta Sequential"
        elif (algorithm == cte.PERCEPTRON):
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
        if (algorithm == cte.PERCEPTRON and i == cte.ITERATIONS - 1):
            plt.show(block=True)
        #else:
         #   plt.show(block=False)
          #  plt.pause(0.005)"""
        
        # Get accuracy
        acc = get_accuracy(x, t, w_aux)
        accuracy.append(acc)
    return accuracy

def plot_data(x, t):
    plt.scatter(x[0,:], x[1,:], c=t[0,:])
    plt.show()

def run_algorithms(x_grid, x, t, w):
    # Case: ∆ Batch
    accuracy_batch = plot_boundary(x_grid, x, t, w, cte.D_BATCH)

    # Case: ∆ Sequential
    accuracy_sequential = plot_boundary(x_grid, x, t, w, cte.D_SEQUENTIAL)

    # Case: Simple Perceptron
    accuracy_perceptron = plot_boundary(x_grid, x, t, w, cte.PERCEPTRON)
    
    return accuracy_batch, accuracy_sequential, accuracy_perceptron
    
"""
def get_accuracy(x, t, w):
    predictions = []
    correct = 0
    
    for i in range(len(t)):        
        prediction = np.dot(w, x[:,i])
        if prediction > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    for j in range(len(predictions)):
        if predictions[j] == t[0][j]:
            correct += 1 
    accuracy = correct / len(t)
    
    return accuracy
    """

def get_accuracy(x, t, w):
    w_aux = w.copy()
    predictions = []
    correct = 0

    for i in range(0, x.shape[1]):
        activation = w_aux[0][len(x[:,i]) - 1]
        for j in range(0, len(x[:,i])):
            activation += w_aux[0][j] * x[:,i][j] # activation is the sum of all the weights * nodes
            
        if (activation >= 0):
            predictions.append(1.0)
        else:
            predictions.append(-1.0)
            
    for j in range(len(predictions)):
        if predictions[j] == t[0][j]:
            correct += 1 
            
    accuracy = correct / t.shape[1]

    return accuracy

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
    dw = (dw * cte.ALPHA) - np.dot(delta_h, np.transpose(x)) * (1 - cte.ALPHA)
    dv = (dv * cte.ALPHA) - np.dot(delta_o, np.transpose(h)) * (1 - cte.ALPHA)
    W = w + dw * cte.ETA
    V = v + dv * cte.ETA
    return W, V

def plot_boundary_multilayer(x, w, v, dw, dv, t, x_grid, y_grid):
    for i in range(cte.ITERATIONS):
        h, o = forward_pass(x, w, v)
        delta_o, delta_h = backward_pass(t, w, v, o, h)
        w, v = weight_update(x, dw, dv, delta_h, delta_o, h, w, v)
        x_input = np.c_[x_grid.ravel(), y_grid.ravel()]
        x_input = np.transpose(x_input)
        x_input = np.vstack((x_input, np.ones((x_input.shape[1]))))
        z = forward_pass(x_input, w, v)[1]
        z = z.reshape(x_grid.shape)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim(min(x[0,:]), max(x[0,:]))
        axes.set_ylim(min(x[1,:]), max(x[1,:]))
        plt.contour(x_grid, y_grid, z, levels=[0])
        plt.title("Multilayer Perceptron: Iteration %i" %(i+1))
        plt.xlabel("$\mathregular{X_1}$ coordinate")
        plt.ylabel("$\mathregular{X_2}$ coordinate")
        plt.scatter(x[0,:], x[1,:], c=t[0,:])
        if (i == 0 or i == cte.ITERATIONS - 1):
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.005)
