import numpy as np


def generate_data():
    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    
    x = np.vstack([x1d, x2d, x3d])
    
    return x

def weights(x):
    N = x.shape[0] # number of patterns
    M = x.shape[1] # number of neurons
    
    w = np.zeros([M, M])
    for i in range(N):
        # calculate weights
        x_i = x[i,:]
        w += np.outer(x_i.T, x_i)
        
    for i in range(M):
        w[i, i] = 0 # fill diagonal with 0

    return w

def update(x, w):
    x_new = w @ x.T
        
    return x_new.T

def recall(x, w):
    N = x.shape[0] # number of patterns
    for i in range(N):
        x_i = np.sum(np.dot(w, x[i,:].reshape(1,-1).T), axis=1)
        assert x[i,:].all() == np.where(x_i > 0, 1, -1).all()
        print('data:  ', x[i,:])
        print('recall:', np.where(x > 0, 1, -1))
    

    


x = generate_data()
w = weights(x)
x_updated = update(x, w)
recall(x_updated, w)