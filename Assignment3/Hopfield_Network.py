import numpy as np 
import matplotlib.pyplot as plt

ITERATIONS = 1 # number of iterations for syncronious update

def generate_data(d_type):
    # For task 3.1
    
    if d_type == "original":
        x1d = [1, -1, 1, -1, 1, -1, -1, 1]
        x2d = [1, 1, -1, -1, -1, 1, -1, -1]
        x3d = [1, 1, 1, -1, 1, 1, -1, 1]
        
    if d_type == "distorted":
        x1d = [1, -1, 1, -1, 1, 1, -1, 1] # one bit error
        x2d = [1, 1, -1, 1, 1, 1, -1, -1] # two bit error
        x3d = [1, 1, 1, -1, 1, -1, 1, 1] # two bit error
    
    if d_type == "super_distorted":
        # more then half the data is distroted
        x1d = [1, 1, -1, 1, 1, 1, 1, -1]
        x2d = [-1, -1, 1, -1, 1, -1, 1, -1]
        x3d = [1, -1, -1, -1, -1, -1, -1, -1]
                
    x = np.vstack([x1d, x2d, x3d])
    
    return x

def weights(x):
    # Update weights for Little Model
    
    N = x.shape[0] # number of patterns
    M = x.shape[1] # number of neurons
    
    w = np.zeros([M, M])
    for i in range(N):
        # calculate weights
        x_i = x[i,:]
        w += np.outer(x_i.T, x_i)
        
    for i in range(M):
        w[i, i] = 0 # fill diagonal with 0

    #w = w / M # normalize, only if bias is used

    return w

def update_syncronously(x, w):
    # Update the weights synchroniously
    # aka "Little Model"
    x_current = np.copy(x)
    x_new = np.copy(x)
    
    # Iterate for convergence
    for _ in range(ITERATIONS):
        #x_new = np.sign(w @ x_current.T) # from 2.2 in lab
        for i in range(x.shape[0]):
            x_new[i, :] = np.sign(x_current[i, :] @ w)
            
        if np.all(x_new == x_current.T): # check recall
            print("The network converged after {} iterations.".format(_))
            break # the state is stable (convergence, break loop)
            
        x_current = x_new
        
    return x_current

def update_asyncronously(x, w):
    # Update the weights synchroniously
    # aka "Little Model"
    x_current = np.copy(x)
    x_new = np.copy(x)
    
    # Iterate for convergence
    for _ in range(ITERATIONS):
        for i in range(x.shape[0]):
            for __ in range(x.shape[1]):
                idx = np.random.randint(0, x.shape[1])
                x_new[i, idx] = np.sign(x_current[i, :] @ w[idx])
            
        if np.all(x_new == x_current.T): # check recall
            print("The network converged after {} iterations.".format(_))
            break # the state is stable (convergence, break loop)
            
        x_current = x_new
        
    return x_current

def display(image):
    # For task 3.2
    
    # display images in shape (32, 32)
    plt.imshow(image.reshape(32,32), interpolation="nearest")
    plt.show()