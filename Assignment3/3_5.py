from Hopfield_Network import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 1
PATTERNS = True
ITERATIVE_WEIGHT = True
DIAGONAL = 'diagonal_0' 
NOISE_ITERATIVE = True

np.random.seed(42)

def random_patterns(units, num_patterns=1):
    return np.array(np.random.choice([-1, 1], size=units * num_patterns)).reshape(-1, units)


def iterative_patterns(w_mat):
    uniques = {}
    for i in range(w_mat.shape[0]):
        aux = ""
        for j in range(len(w_mat[i, :])):
            if j != len(w_mat[i, :]) - 1:
                aux += str(w_mat[i, j]) + " "
            else:
                aux += str(w_mat[i, j])
        if aux not in uniques:
            uniques[aux] = 0

    return w_mat.shape[0] - len(uniques)  # Returning the number of not unique vectors


def main():
    # Load data
    data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
    num_units = 50
    num_patterns = 150
    patterns = random_patterns(num_units, num_patterns)

    if PATTERNS:
        if ITERATIVE_WEIGHT:
            uniques = list()
            counter = np.zeros(num_patterns)
            for i in tqdm(range(num_patterns)):
                w = weights(patterns[:i+1, :].reshape(i+1, num_units), diagonal = DIAGONAL)
                for j in range(i+1):
                    if NOISE_ITERATIVE:
                        counter_aux, x_current = noised_images([0], patterns, j, counter, w, return_data=True, iterative_patterns=NOISE_ITERATIVE)
                    else: 
                        x_current = recall(patterns[j:j+1, :], w, update_type="synchronous",
                                   convergence_type='energy')
                    if iterative_patterns:
                        counter[i] += iterative_patterns_accuracy(patterns[:i+1, :], x_current)
            plt.figure()
            plt.plot(counter)   
            plt.show()
        else:
            w = weights(patterns, diagonal = DIAGONAL)
            image = 0
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, patterns, image, counter, w, noised_iterations=ITERATIONS)
            plt.figure()
            # sns.set_style('darkgrid')
            counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
            counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
            plt.show()
        

    else:
        for pat in range(3, 8):
            print("Running network for image", pat)
            w = weights(data[:pat, :], diagonal = DIAGONAL)
            image = 0
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, data, image, counter, w, noised_iterations=ITERATIONS)
            plt.figure()
            # sns.set_style('darkgrid')
            counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
            counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
            plt.show()
            input("Press enter to continue...")


if __name__ == "__main__":
    main()
