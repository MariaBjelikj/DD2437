import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers 
from sklearn.metrics import mean_squared_error as mse
import Constants as cte
import time

def best_validation_score(layers, train_input, train_output, validation_input, validation_output, test_input, test_output):
    
    NN = neural_network(layers)
    early_stop_count = 0
    validation_MSE = [] # to store all MSE
    flag = True

    for i in range(10):
        if (flag == False): 
            break
        for iter in range(cte.ITERATIONS):
            history = NN.fit(train_input, train_output, batch_size=50, verbose=0)
            prediction = NN.predict(validation_input, verbose=0)
            validation_MSE.append(mse(validation_output, prediction))
            
            # After 2 iterations, we can start comparing
            if (iter > 2):                
                # Check tolerence, break loop to early stop
                if (validation_MSE[-2] - validation_MSE[-1] < cte.EARLY_STOP_TOLERANCE):
                    # Check early stop threshold 
                    if (early_stop_count < cte.EARLY_STOP_THRESHOLD):
                        early_stop_count += 1
                    else:
                        break
                else:
                    early_stop_count = 0
        # Evaluate test data and check success
        if (NN.evaluate(test_input, test_output) > 0.5):
            flag = False
            
    return history, NN, validation_MSE

def x_previous(t, x):
    if (t < 0):
        return 0
    return x[t]
    
def x_current(t, x, beta, gamma, n, tau):
    return x_previous(t - 1, x) + (beta * x_previous(t - tau - 1, x)) / (1 + x_previous(t - tau - 1, x) ** n) - gamma * x_previous(t - 1, x)

def generate_data():
    """
    Generates Mackey-Glass data as:
        
    x(t+1) = x(t) + (0.2 * x(t - 25)) / (1 + x^10(t - 25)) - 0.1 * x(t)
    x(0) = 1.5
    x(t) = 0 for t < 0
    """
    beta = 0.2
    gamma = 0.1
    n = 10
    tau = 25
    N =  1600
    x = np.zeros(N)
    x[0] = 1.5

    for t in range(1, N):
        x[t] = x_current(t, x, beta, gamma, n, tau)

    return x

def neural_network(layers):
    
    num_layers = len(layers)
    model = Sequential()

    # Add first layer
    model.add(Dense(layers[0], input_dim=5, activation='relu', kernel_regularizer=regularizers.l2(cte.LAMBDA), use_bias=True))
    # Add more layers
    for i in range(1, num_layers-1):
        model.add(Dense(layers[i], activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(cte.LAMBDA)))
    
    model.add(Dense(layers[num_layers-1], activation='relu', use_bias=True))

    model.summary()
    model.compile(optimizer="adam", loss='mse')
    #early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=100)
    
    return model


def main():
    """
    Data: 
    t = 301:1500;
    input = [x(t - 20); x(t - 15); x(t - 10); x(t - 5); x(t)];
    output = x(t + 5);
    """
    x = generate_data()
    t = np.arange(301, 1501)
    
    Input = np.transpose(np.array([x[t - delay] for delay in range(-20, 1, 5)]))
    output = x[t + 5]
    
    train_input = Input[:-200]
    if (cte.NOISE_SIGMA != 0):
        for i in range(len(train_input)):
            train_input[i] += np.random.normal(0, cte.NOISE_SIGMA)
    train_output = output[:-200]
    validation_input = Input[-500:-200] 
    if (cte.NOISE_SIGMA != 0):
        for i in range(len(validation_input)):
            validation_input[i] += np.random.normal(0, cte.NOISE_SIGMA)
    validation_output = output[-500:-200]
    test_input = Input[-200:] 
    test_output = output[-200:]
    
    mse_aux = float("-inf")
    best_1= 0
    best_2= 0
    """
    for i in range(1,9):
        for j in range(1,2):
        # Different network configurations
            layers = [i, j, 1]

            history, NN, validation_MSE = best_validation_score(layers, train_input, train_output,\
                            validation_input, validation_output,\
                            test_input, test_output)
        
            # Get best validation score
            best_validation_MSE = validation_MSE[-1]
            mse_validation = best_validation_MSE
            
            # Prediction on test data
            
            prediction = NN.predict(test_input, verbose=0)
            mse_pred = mse(test_output, prediction)
            
            if mse_aux < mse_validation:
                mse_aux = mse_validation
                best_1 = i
                best_2 = j
    """
    print("best hidden layer size combination is:", best_1, best_2)
    layers = [8, 8, 1]

    history, NN, validation_MSE = best_validation_score(layers, train_input, train_output,\
                        validation_input, validation_output,\
                        test_input, test_output)

    # Get best validation score
    best_validation_MSE = validation_MSE[-1]
    mse_validation = best_validation_MSE
    
    # Prediction on test data
    
    prediction = NN.predict(test_input, verbose=0)
    mse_pred = mse(test_output, prediction)
    
    print("Validation MSE for the best layer:", mse_validation)
    print("Test MSE for the best layer:", mse_pred)

    x_grid = np.arange(len(train_output) + len(validation_output), len(train_output) + len(validation_output) + len(test_output))
    predicted_plot = prediction
    plt.plot(x_grid, test_output, label="Test data", color="Orange")
    plt.plot(x_grid, predicted_plot, markersize=2, label="Predicted data", color="#3185ac")
    plt.fill_between(x_grid, predicted_plot.ravel() + cte.NOISE_SIGMA, predicted_plot.ravel() - cte.NOISE_SIGMA, alpha=0.5, color="#51baea", edgecolor="#3185ac", label="Confidence interval")
    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    plt.title("Test data targets vs. prediction")
    plt.legend()
    plt.show()
    
    # Histogram of weights
    weights, biases = NN.layers[0].get_weights()
    plt.hist(weights)
    plt.xlabel("Value of weights")
    plt.ylabel("Weights")
    plt.title("Weight distribution for {} units, lambda = {}".format(layers[0], cte.LAMBDA))
    plt.show()
    
    # TODO: different regularizations?

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
