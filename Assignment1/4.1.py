import numpy as np
import matplotlib.pyplot as plt

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
    N = 2000

    x = np.zeros(N)
    x[0] = 1.5

    for t in range(1, N):
        x[t] = x_current(t, x, beta, gamma, n, tau)

    return x

def main():
    """
    t = 301:1500;
    input = [x(t - 20); x(t - 15); x(t - 10); x(t - 5); x(t)];
    output = x(t + 5);
    """
    x = generate_data()
    t = np.arange(301, 1501)
    
    for i in range(3):
        for delay in range(-20, 1, 5):
            input = np.transpose(np.array([i, x[t - delay]]))
            output = x[t + 5]
    
    train_input = input[:-500]
    train_output = output[:-500]
    validation_input = input[-500:-200] 
    validation_output = output[-500:-200]
    test_input = input[-200:] 
    test_output = output[-200:]

    plt.plot(np.arange(len(train_output)), train_output, label="Training data")
    plt.plot(np.arange(len(train_output), len(train_output) + len(validation_output)), validation_output, label="Validation data")
    plt.plot(np.arange(len(train_output) + len(validation_output), len(train_output) + len(validation_output) + len(test_output)), test_output, label="Test data")
    plt.legend()
    plt.title("Data")
    plt.show()



if __name__ == "__main__":
    main()