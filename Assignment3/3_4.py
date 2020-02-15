from Hopfield_Network import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 10
IMAGE = [0, 1, 2]

np.random.seed(42)


def main():
    # Load data
    data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

    w = weights(data[:3, :])
    for image in tqdm(IMAGE):
        counter = np.zeros(len(PERCENTAGES))
        counter = noised_images(PERCENTAGES, data, image, counter, w, noised_iterations=ITERATIONS)
        plt.plot(PERCENTAGES, counter, label="Image {}".format(image))
    plt.xlabel('Noise level')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # PLOT SPECIFIC PERCENTAGE
    # counter = np.zeros(len(PERCENTAGES))
    # counter, x_current = noised_images([0.9], data, image, counter, w,
    # noised_iterations=ITERATIONS, return_data = True)
    # display(x_current, title = "Recovered noised imaged with error 0.9")


if __name__ == "__main__":
    main()
