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


def recall(x, w, update_type="synchronous", convergence_type=False):
    if update_type == "synchronous":
        # Update the weights synchronously, aka "Little Model"
        x_current = np.copy(x)
        x_new = np.copy(x)
        
        # Compute energy of the initial state
        E_current = energy(x, w)
        convergence_count = 0
        
        # Iterate for convergence
        for iteration in range(ITERATIONS):
            for i in range(x.shape[0]):
                x_new[i, :] = set_sign(x_current, w, i)
                # Compute energy for this state
            
            if convergence_type == "energy":
                E_new = energy(x_new, w)
                    
                if E_new == E_current: # If the energy hasn't changed, converged
                    convergence_count += 1
                    
                if convergence_count > 5:
                    print("The network converged after {} iterations.".format(iteration))
                    break
                    
                E_current = np.copy(E_new)
            
            else: 
                if np.all(x_new == x_current.T):  # check recall
                    print("The network converged after {} iterations.".format(iteration))
                    break  # the state is stable (convergence, break loop)
                
            x_current = np.copy(x_new)
    
    if update_type == "asynchronous":
        # Update the weights asynchroniously
        x_current = np.copy(x)
        x_new = np.copy(x)
        
        # Compute energy of the initial state
        E_current = energy(x, w)
        convergence_count = 0
 
        # Iterate for convergence
        for iteration in range(ITERATIONS):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    idx = np.random.randint(0, x.shape[1])
                    x_new[i, idx] = np.sign(x_current[i, :] @ w[idx])
                    
            if convergence_type == "energy":
                E_new = energy(x_new, w)
                    
                if E_new == E_current: # If the energy hasn't changed, converged
                    convergence_count += 1
                    
                if convergence_count > 5:
                    print("The network converged after {} iterations.".format(iteration))
                    break
                    
                E_current = np.copy(E_new)
    
            else:
                if np.all(x_new == x_current.T):  # check recall
                    print("The network converged after {} iterations.".format(iteration))
                    break  # the state is stable (convergence, break loop)
                    
            x_current = np.copy(x_new)
                          
    return x_current

def find_attractors(data, data_updated, w):
    # Attractors are the patterns that don't change after weight update
    attractors = []
    
    for i in range(data.shape[0]):
        if np.all(data[i] == data_updated[i]):
            attractors.append(data[i])
    
    return attractors
        

def energy(state, w):
    return np.sum(- 0.5 * state @ w @ state.T)

def display(image):
    # For task 3.2

    # display images in shape (32, 32)
    plt.imshow(image.reshape(32, 32), interpolation="nearest")
    plt.show()
    
    
    
# TODO: Add bias
# Asynchronous update: Maybe we should change the order of how we update?