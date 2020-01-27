import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def generate_data(type, noise):
    x_train = np.arange(0, 2 * np.pi, 0.1)[:,np.newaxis]
    x_test = np.arange(0.05, 2 * np.pi, 0.1)[:,np.newaxis]
    
    if type == sin2x:
        y_train = np.sin(2 * x_train)
        y_test = np.sin(2 * x_test)
            
    if type == square:
        y_train = sign(np.sin(2 * x_train))
        y_test = sign(np.sin(2 * x_test))
    
def phi(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance ** 2))
    
def train_batch(x, y, mean, variance):
    phi = phi(x, mean, variance)
    
    

def main():
    

if __name__ == "__main__":
    main()