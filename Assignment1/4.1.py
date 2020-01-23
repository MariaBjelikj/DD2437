import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras import regularizers 
from sklearn.metrics import mean_squared_error as mse

def x_previous(t, x):
    if t < 0: return 0
    else: return x[t]
    
def x_current(t, x, beta, gamma, n, tau):
    return x_previous(t - 1, x) + (beta * x_previous(t - tau - 1, x)) / (1 + x_previous(t - tau - 1, x) ** n) - gamma * x_previous(t - 1, x)

def generate_data():
    """
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

def neural_network(layers, input_dim):
    
    num_layers = len(layers)
    model = Sequential()

    # Add first layer
    model.add(Dense(layers[0], input_dim=input_dim, kernel_initializer='normal',\
                    activation='relu', use_bias=True))
    
    # Add more layers
    for i in range(1, num_layers):
        model.add(Dense(layers[i], kernel_initializer='normal', activation='relu', use_bias=True))

    model.summary()
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=100)
    
    return model, early_stop


def main():
    """
    t = 301:1500;
    input = [x(t - 20); x(t - 15); x(t - 10); x(t - 5); x(t)];
    output = x(t + 5);
    """
    x = generate_data()
    t = np.arange(301, 1501)
    
    Input = np.transpose(np.array([x[t - delay] for delay in range(-20, 1, 5)]))
    output = x[t + 5]
    
    train_input = Input[:-200]
    train_output = output[:-200]
    #validation_input = Input[-500:-200] 
    #validation_output = output[-500:-200]
    test_input = Input[-200:] 
    test_output = output[-200:]
    
    # Plot data
    """plt.plot(np.arange(len(train_output)), train_output, label="Training data")
    plt.plot(np.arange(len(train_output), len(train_output) + len(validation_output)), validation_output, label="Validation data")
    plt.plot(np.arange(len(train_output) + len(validation_output), len(train_output) + len(validation_output) + len(test_output)), test_output, label="Test data")
    plt.legend()
    plt.title("Data")
    plt.show()"""
    
    layers = [4, 1]
    input_dim = 5
    
    NN, early_stop = neural_network(layers, input_dim)
    
    history = NN.fit(train_input, train_output, epochs=1000, \
                        validation_split=0.3, batch_size=50, verbose=1)
 
    # TODO: callbacks for early stop

    #MSE = NN.evaluate(validation_input, validation_output, verbose=0) 
    #print('MSE is. {}'.format(MSE))


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Trainining and validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()