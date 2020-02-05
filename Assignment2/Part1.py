import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

EPOCHS = 100
ETA = 0.1
ETA_list = [0.007, 0.01, 0.05, 0.1, 0.2]
ITERATIONS = 1000  # for CL


def generate_data(f_type, noise=False):
    x_train, y_train, x_test, y_test = 0, 0, 0, 0
    # Create y as a sine function
    if f_type == "sin2x":
        x_train = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]  # For training data the range is 0 to 2pi
        y_train = np.sin(2 * x_train)
        x_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]  # Test data starts from 0.05
        y_test = np.sin(2 * x_test)

    # Create y square
    if f_type == "square2x":
        x_train = np.arange(0, 2 * np.pi, 0.1)[:, np.newaxis]  # For training data the range is 0 to 2pi
        y_train = np.where(np.sin(2 * x_train) >= 0, 1, -1)
        x_test = np.arange(0.05, 2 * np.pi, 0.1)[:, np.newaxis]  # Test data starts from 0.05
        y_test = np.where(np.sin(2 * x_test) >= 0, 1, -1)

    # Load ballist and balltest and splits the inputs x and outputs y
    if f_type == "ballist":
        train = np.loadtxt("data_lab2/ballist.dat")
        test = np.loadtxt("data_lab2/balltest.dat")

        x_train = train[:, :2]
        y_train = train[:, 2:]
        x_test = test[:, :2]
        y_test = test[:, 2:]

    # Add noise for 3_2, zero mean data with variance 0.1
    # no noise for ballist data
    if noise and f_type != "ballist":
        for i in range(y_train.shape[0]):
            y_train[i] = y_train[i] + np.random.normal(0, 0.1)
        for i in range(y_test.shape[0]):
            y_test[i] = y_test[i] + np.random.normal(0, 0.1)

    return x_train, y_train, x_test, y_test


def phi_matrix(x, mean, variance):
    # Create phi matrix where every element is calculated as the RBF
    # For each data sample and for each RBF
    phi = np.zeros((x.shape[0], mean.shape[0]))
    for i in range(x.shape[0]):
        for j in range(mean.shape[0]):
            # RBF phi as specified in (1)
            phi[i][j] = np.sum(np.exp(-(x[i] - mean[j]) ** 2 / (2 * variance ** 2)))

    return phi


def train_batch(x, y, mean, variance):
    phi = phi_matrix(x, mean, variance)

    # Find w by solving the system in (4)
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi)), np.dot(phi.T, y))

    return w


def train_online_delta_rule(x, y, mean, variance):
    phi = phi_matrix(x, mean, variance)
    w = np.random.rand(1, phi.shape[1])  # random initialisation for weights

    # for ETA in ETA_list:
    mse_list = []
    mae_list = []
    for epoch in range(EPOCHS):
        for i in range(phi.shape[0]):
            phi_i = np.reshape(phi[i], (1, len(phi[i])))  # just to avoid dimension error
            error = y[i] - np.dot(phi_i, w.T)  # error from real value to predicted
            delta_w = ETA * np.dot(phi_i.T, error)  # equation 10
            w += delta_w.T

        y_predicted = np.dot(phi, w.T)

        mse_val = mse(y, y_predicted)
        mae_val = mae(y, y_predicted)

        if epoch > 1:
            mse_list.append(mse_val)
            mae_list.append(mae_val)

        # plt.plot(range(2, EPOCHS), MAE, label=ETA)

    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("MAE")
    # plt.yscale('log')
    # plt.show()

    """# Plot MAE vs Epochs
    plt.plot(range(2, EPOCHS), MAE)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()"""

    return w.T


def predict(x, mean, variance, w):
    phi = phi_matrix(x, mean, variance)

    # The prediction is done multiplying RBF phi with the calculated weights w
    return np.dot(phi, w)


def predict_square(x, y, mean, variance, w):
    # Use this function to transform the error to 0.0
    phi = phi_matrix(x, mean, variance)

    prediction = np.ones(y.shape)
    prediction[np.where(np.dot(phi, w) >= 0)] = 1
    prediction[np.where(np.dot(phi, w) < 0)] = -1

    return prediction


def competitive_learning(x, deadnode=False):
    np.random.shuffle(x)  # Shuffle data for more random selection
    rbf_nodes = x[[0, 3, 8, 15, 22, 29, 34, 37, 45, 52, 58, 62]]  # Select nodes from data
    # the higher the number of RBF nodes here, the better the prediction

    # plt.scatter(RBF_nodes, np.zeros(len(RBF_nodes)), color='r', label="RBF nodes before")

    for i in range(ITERATIONS):
        train_vector = x[np.random.randint(0, len(x)), :]  # random vector for training
        distances = []

        for node in rbf_nodes:
            # Calculate distance between training vector and RBF node
            distances.append(np.linalg.norm(node - train_vector))

        # Get indices to sort the nodes (this gives index 0 to the smallest distance,
        # 1 to the next smallest, etc. and index n for the largest distance)
        index = np.argpartition(distances, 1)

        winner = np.argmin(index)  # get best node (smallest distance)
        rbf_nodes[winner] += ETA * (train_vector - rbf_nodes[winner])

        if deadnode:
            # Update the worst node (largest distance) to avoid dead nodes
            loser = np.argmax(index)
            rbf_nodes[loser] += ETA * ETA * (train_vector - rbf_nodes[loser])
            # NOTE: another option here is to select multiple winners!

    # plt.scatter(RBF_nodes, np.zeros(len(RBF_nodes)), color='b', label="RBF nodes after")
    # plt.title("Change of RBF nodes using Competitive Learning")
    # plt.legend()
    # plt.show()

    return rbf_nodes


def main():
    # --------------- 3.1: Training in Batch --------------- #
    error_thresholds = [0.1, 0.01, 0.001]
    f_type = "sin2x"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    y_predicted = 0
    rbf_nodes = 19

    # variance_list = [0.1, 0.5, 0.8, 1.2, 1.5]

    # Test error thresholds for various numbers of RBF nodes
    # for variance in variance_list:
    #    MAE = [] # Reset MAE for new width
    for i in range(1, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 0.5  # remove for loop and uncomment this for fixed width
        w = train_batch(x_train, y_train, mean, variance)

        if f_type == "sin2x":
            y_predicted = predict(x_test, mean, variance, w)
        else:
            y_predicted = predict_square(x_test, y_test, mean, variance, w)
        # y_predicted = predict(x_test, y_test, mean, variance, w)
        # y_predicted = predict(x_test_clean, y_test_clean, mean, variance, w)

        # mean_square_error = mse(y_test, y_predicted)
        # print("The MSE is: {}".format(mean_square_error))
        mean_absolute_error = mae(y_test, y_predicted)
        # print("The absolute residual error is {} for {} nodes".format(mean_absolute_error, i))

        # Check if the error is lower than the error thresholds
        # We want to check against the largest threshold
        if len(error_thresholds) and mean_absolute_error < max(error_thresholds):
            # Threshold fulfilled, remove
            print("Threshold {} removed for {} nodes.".format(max(error_thresholds), i))
            print("Obtained error was {}".format(mean_absolute_error))
            error_thresholds.remove(max(error_thresholds))

        # MAE.append(mean_absolute_error)
        # plt.plot(range(1, rbf_nodes), MAE, label=variance)

    # plt.legend()
    # plt.xlabel("Number of RBF nodes")
    # plt.ylabel("MAE")
    # plt.title("Batch mode: Absolute residual error vs. number of nodes \n for different RBF widths given in legend.")
    # plt.show()

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.title("Training in batch mode")
    plt.legend()
    plt.show()

    # --------------- 3.2: Online Delta Rule --------------- #
    f_type = "sin2x"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    x_train_clean, y_train_clean, x_test_clean, y_test_clean = generate_data(f_type, False)
    y_predicted = 0
    rbf_nodes = 20

    # for variance in variance_list:
    #    MAE = [] # Reset MAE for new width
    for i in range(1, rbf_nodes):
        mean = np.linspace(0, 2 * np.pi, i)
        variance = 1  # remove for loop and uncomment this for fixed width

        w = train_online_delta_rule(x_train, y_train, mean, variance)

        # if f_type == "sin2x": y_predicted = predict(x_test, y_test, mean, variance, w)
        # else: y_predicted = predict_square(x_test, y_test, mean, variance, w)
        # y_predicted = predict(x_test, y_test, mean, variance, w)
        y_predicted = predict(x_test_clean, mean, variance, w)

        # mean_square_error = mse(y_test, y_predicted)
        # print("The MSE is: {}".format(MSE))
        # mean_absolute_error = mae(y_test, y_predicted)
        # mean_absolute_error = mae(y_test_clean, y_predicted)
        # print("The absolute residual error is {} for {} nodes".format(mean_absolute_error, i))

        # MAE.append(mean_absolute_error)

        # one tab on everything to use for loop for variance
        # plt.plot(range(1, rbf_nodes), MAE, label=variance)

    # plt.legend()
    # plt.xlabel("Number of RBF nodes")
    # plt.ylabel("MAE")
    # plt.title("Online Delta: Absolute residual error vs. number of nodes
    # \n for different RBF widths given in legend.")
    # plt.show()

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.title("Online Delta Rule")
    plt.legend()
    plt.show()

    # --------------- Single Hidden Layer Perceptron --------------- #

    # --------------- 3.3: Competitive Learning --------------- #

    # --------------- Training in Batch --------------- #
    f_type = "sin2x"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    # y_predicted = 0
    variance = 0.5
    rbf_nodes = competitive_learning(x_train.copy())

    mean = rbf_nodes
    w = train_batch(x_train, y_train, mean, variance)

    # if f_type == "sin2x": y_predicted = predict(x_test, y_test, mean, variance, w)
    # else: y_predicted = predict_square(x_test, y_test, mean, variance, w)
    y_predicted = predict(x_test, mean, variance, w)
    # y_predicted = predict(x_test_clean, y_test_clean, mean, variance, w)

    # mean_square_error = mse(y_test, y_predicted)
    # print("The MSE is: {}".format(mean_square_error))
    mean_absolute_error = mae(y_test, y_predicted)
    print("The absolute residual error is {}".format(mean_absolute_error))

    plt.plot(x_train, y_train, label='real output')
    plt.plot(x_test, y_predicted, 'r--', label='prediction')
    plt.title("Training in batch mode, competitive learning")
    plt.legend()
    plt.show()

    # --------------- Ballist data --------------- #
    f_type = "ballist"
    x_train, y_train, x_test, y_test = generate_data(f_type, True)
    # y_predicted = 0
    variance = 0.9
    rbf_nodes = competitive_learning(x_train.copy())
    print(rbf_nodes)

    # Train the RBF network in batch mode
    mean = rbf_nodes
    w = train_batch(x_train, y_train, mean, variance)
    y_predicted = predict(x_test, mean, variance, w)

    mean_absolute_error = mae(y_test, y_predicted)
    print("The absolute residual error is {}".format(mean_absolute_error))

    """plt.plot(y_predicted[:,0], y_predicted[:,1], "x", color="r", label="prediction")
    plt.plot(y_test[:,0], y_test[:,1], "o", color="g", label="real output")
    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.legend()
    plt.title("Ballist data: Training in batch mode, competitive learning")
    plt.show()"""


if __name__ == "__main__":
    main()
