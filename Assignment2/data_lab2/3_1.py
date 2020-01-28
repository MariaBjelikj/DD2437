import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

sign = lambda x: np.where(x >= 0, 1, -1)

def generate_data(type, noise=False):
    x_train = np.arange(0, 2 * np.pi, 0.1)[:,np.newaxis]
    x_test = np.arange(0.05, 2 * np.pi, 0.1)[:,np.newaxis]
    
    if type == "sin2x":
        y_train = np.sin(2 * x_train)
        y_test = np.sin(2 * x_test)
            
    if type == "square":
        y_train = sign(np.sin(2 * x_train))
        y_test = sign(np.sin(2 * x_test))
        
    # TODO: Add noise for 3_2
    return x_train, y_train, x_test, y_test
    

def Phi(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance ** 2))

    
def train_batch(x, y, mean, variance):
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i][j] = Phi(x[i], mean[j], variance)
            
    # Find w by solving the system in (4)
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi)), np.dot(phi.T, y))
    
    return w

def predict(x, y, mean, variance, w):
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i][j] = Phi(x[i], mean[j], variance)
    
    return np.dot(phi, w)

def main():
    
    error_thresholds = [0.1, 0.01, 0.001]
    x_train, y_train, x_test, y_test = generate_data("sin2x")
    
    rbf_nodes = 20
    
    for i in range(5, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 1
        
        w = train_batch(x_train, y_train, mean, variance)
        
        phi = np.zeros((x_test.shape[0], mean.shape[0]))
        for i in range(x_test.shape[0]):
            for j in range(mean.shape[0]):
                phi[i][j] = Phi(x_test[i], mean[j], variance)
        
        y_predicted = predict(x_test, y_test, mean, variance, w)
        
        MSE = mse(y_test, y_predicted)
        #print("The MSE is: {}".format(MSE))
        
        MAE = mae(y_test, y_predicted)
        #print("The mean absolute error is: {}".format(MAE))
        
        if len(error_thresholds) and MAE < max(error_thresholds):
            # Threshold fulfilled, remove
            error_thresholds.remove(max(error_thresholds))
            
    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()