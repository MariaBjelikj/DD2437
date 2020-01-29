import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

EPOCHS = 100
ETA = 0.1


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

    # Add noise for 3_2, zero mean data with variance 0.1
    if noise:
        y_train = y_train + np.random.normal(0, 0.1, x_train.shape[1])
        y_test = y_test + np.random.normal(0, 0.1, x_test.shape[1])

    return x_train, y_train, x_test, y_test


def phi_calculation(x, mean, variance):
    # RBF phi as specified in (1)
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


def train_online_delta_rule(x, y, mean, variance):
    phi = phi_matrix(x, mean, variance)
    w = np.random.rand(1, phi.shape[1])  # random initialisation for weights
    mse_list = []

    for epoch in range(EPOCHS):
        for i in range(phi.shape[0]):
            phi_i = np.reshape(phi[i], (1, len(phi[i])))  # just to avoid dimension error
            error = y[i] - np.dot(phi_i, w.T)  # error from real value to predicted
            delta_w = ETA * np.dot(phi_i.T, error)  # equation 10
            w += delta_w.T

        y_predicted = np.dot(phi, w.T)

        mse_ = mse(y, y_predicted)

        if epoch > 1:
            mse_list.append(mse_)

    # Plot MSE vs Epochs
    plt.plot(range(2, EPOCHS), mse_list)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()

    return w


def predict(x, mean, variance, w):
    phi = phi_matrix(x, mean, variance)

    # The prediction is done multiplying RBF phi with the calculated weights w
    return np.dot(phi, w)


def predict_square(x, y, mean, variance, w):
    phi = phi_matrix(x, mean, variance)

    prediction = np.ones(y.shape)
    prediction[np.where(np.dot(phi, w) >= 0)] = 1
    prediction[np.where(np.dot(phi, w) < 0)] = -1

    return prediction


def main():
    #  3.1: Training in Batch  #
    error_thresholds = [0.1, 0.01, 0.001]
    f_type = "sin2x"
    x_train, y_train, x_test, y_test = generate_data(f_type, False)
    y_predicted = 0
    rbf_nodes = 20

    # Test error thresholds for various numbers of RBF nodes
    for i in range(5, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 1
        w = train_batch(x_train, y_train, mean, variance)

        if f_type == "sin2x":
            y_predicted = predict(x_test, y_test, mean, variance, w)
        else:
            y_predicted = predict_square(x_test, y_test, mean, variance, w)

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
    plt.title("Training in batch mode")
    plt.legend()
    plt.show()

    #  3.2: Online Delta Rule  #
    f_type = "sin2x"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    y_predicted = 0
    rbf_nodes = 50

    mean = np.linspace(0, 2 * np.pi, rbf_nodes)
    variance = 1

    w = train_online_delta_rule(x_train, y_train, mean, variance)

    if f_type == "sin2x":
        y_predicted = predict(x_test, mean, variance, w.T)
    else:
        y_predicted = predict_square(x_test, y_test, mean, variance, w.T)

    mean_square_error = mse(y_test, y_predicted)
    # print("The MSE is: {}".format(MSE))
    mean_absolute_error = mae(y_test, y_predicted)
    # print("The mean absolute error is: {}".format(MAE))

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.title("Online Delta Rule")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
