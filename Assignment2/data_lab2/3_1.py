import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def generate_data(f_type, noise=False):
    # For training data the range is 0 to 2pi
    x_train = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    
    # Test data starts from 0.05
    x_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
    y_train = 0
    y_test = 0

    # Creating y as a sine function
    if f_type == "sin2x":
        y_train = np.sin(2 * x_train)
        y_test = np.sin(2 * x_test)

    # Creating y square
    if f_type == "square":
        y_train = np.where(np.sin(2 * x_train) >= 0, 1, -1)
        y_test = np.where(np.sin(2 * x_test) >= 0, 1, -1)
        
    # Add noise for 3_2
    if noise:
        y_train = y_train + np.random.normal(0, 0.1, x_train.shape[1])
        y_test = y_test + np.random.normal(0, 0.1, x_test.shape[1])

    return x_train, y_train, x_test, y_test


def phi_calculation(x, mean, variance):
    # RBF phi
    return np.exp(-(x - mean) ** 2 / (2 * variance ** 2))


def phi_matrix(x, mean, variance):
    # Create phi matrix where every element is calculates as the RBF 
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            phi[i][j] = phi_calculation(x[i], mean[j], variance)

    return phi


def train_batch(x, y, mean, variance):
    phi = phi_matrix(x, mean, variance)
    # Find w by solving the system in (4)
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi)), np.dot(phi.T, y))

    return w


def predict(x, y, mean, variance, w):
    phi = phi_matrix(x, mean, variance)
    
    # The prediction is done multiplying RBF phi with the calculated weights w
    return np.dot(phi, w)

def predict_square(x, y, mean, variance, w):
    phi = phi_matrix(x, mean, variance)
    
    prediction = np.ones(y.shape)
    prediction[np.where(np.dot(phi, w) >= 0)] = 1
    prediction[np.where(np.dot(phi, w)  < 0)] = -1
    
    return prediction


def main():
    error_thresholds = [0.1, 0.01, 0.001]
    f_type = "square"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    y_predicted = 0
    rbf_nodes = 20
    
    print(y_train)
    
    # Test error thresholds for various numbers of RBF nodes
    for i in range(5, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 1
        w = train_batch(x_train, y_train, mean, variance)
        phi = phi_calculation(x_test, mean, variance)
        
        if f_type == "sin2x":
            y_predicted = predict(x_test, y_test, mean, variance, w)
        else: y_predicted = predict_square(x_test, y_test, mean, variance, w)
        
        mean_square_error = mse(y_test, y_predicted)
        # print("The MSE is: {}".format(MSE))
        mean_absolute_error = mae(y_test, y_predicted)
        # print("The mean absolute error is: {}".format(MAE))

        # Check if the error is lower than the error thresholds
        # We want to check against the largest threshold
        if len(error_thresholds) and mean_absolute_error < max(error_thresholds):
            # Threshold fulfilled, remove
            error_thresholds.remove(max(error_thresholds))

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
