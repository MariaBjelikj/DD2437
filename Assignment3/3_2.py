from Hopfield_Network import *
import numpy as np


def main():
    # Load data
    data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

    # Learn the first 3 patterns
    x = data[:3, :].copy()
    w = weights(x)

    """prediction = recall(data[9:11, :], w, update_type="asynchronous")
    
    print("Trying to complete degraded pattern p10.")
    display(data[9:10, :])
    display(prediction[0])
    # This degraded pattern converges to stable pattern p1 (pattern data[0])
    #display(data[0])
    
    print("Trying to complate degraded pattern p11.")
    display(data[10:11, :])
    display(prediction[1])"""

    # Random asynchronous (uncomment code for 3.2 in Hopfield_Network)
    display(data[10])
    recall(data[10:11, :].copy(), w, update_type="asynchronous", asyn_type='random')


if __name__ == "__main__":
    main()
