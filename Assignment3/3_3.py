from Hopfield_Network import *


def main():
    # Load data
    data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    # Learn the first 3 patterns
    x = data[:3, :].copy()
    w = weights(x, weights_type="normal")
    prediction = recall(data[10:11, :], w, update_type="asynchronous")
    display(prediction)


if __name__ == "__main__":
    main()
