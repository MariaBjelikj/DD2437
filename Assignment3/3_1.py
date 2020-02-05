import numpy as np

ITERATIONS = 100000 # number of iterations for syncronious update

def generate_data(d_type):
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


def update_syncroniously(x, w):
    # Update the weights synchroniously
    # aka "Little Model"
    x_current = np.copy(x)
    
    for _ in range(ITERATIONS):
        x_new = np.sign(w @ x_current.T) # from 2.2 in lab
        if np.all(x_new == x_current.T): # check recall
            print("The network converged after {} iterations.".format(_))
            break # the state is stable (convergence, break loop)
        x_current = x_new.T
        
    return x_current.astype(int)
            


x = generate_data("original")
x_distorted =  generate_data("distorted")
x_super_distorted =  generate_data("super_distorted")

w = weights(x)
x_updated = update_syncroniously(x, w)
x_updated_distorted = update_syncroniously(x_distorted, w)
x_updated_super_distorted = update_syncroniously(x_super_distorted, w)
