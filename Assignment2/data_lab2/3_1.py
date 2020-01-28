import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def generate_data(f_type, noise=False):
    x_train = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]
    x_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]
    y_train = 0
    y_test = 0

    if f_type == "sin2x":
        y_train = np.sin(2 * x_train)
        y_test = np.sin(2 * x_test)

    if f_type == "square":
        y_train = np.where(np.sin(2 * x_train) >= 0, 1, -1)
        y_test = np.where(np.sin(2 * x_test) >= 0, 1, -1)

    # TODO: Add noise for 3_2
    return x_train, y_train, x_test, y_test


def phi_calculation(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance ** 2))


def phi_matrix(x, mean, variance):
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

    return np.dot(phi, w)


def main():
    error_thresholds = [0.1, 0.01, 0.001]
    x_train, y_train, x_test, y_test = generate_data("sin2x")
    y_predicted = 0
    rbf_nodes = 20

    for i in range(5, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 1
        w = train_batch(x_train, y_train, mean, variance)
        phi = phi_calculation(x_test, mean, variance)
        y_predicted = predict(x_test, y_test, mean, variance, w)
        mean_square_error = mse(y_test, y_predicted)
        # print("The MSE is: {}".format(MSE))
        mean_absolute_error = mae(y_test, y_predicted)
        # print("The mean absolute error is: {}".format(MAE))

        if len(error_thresholds) and mean_absolute_error < max(error_thresholds):
            # Threshold fulfilled, remove
            error_thresholds.remove(max(error_thresholds))

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
