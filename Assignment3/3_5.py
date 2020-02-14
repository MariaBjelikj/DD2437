from Hopfield_Network import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 100
PATTERNS = True
BIASED_PATTERNS = True
ITERATIVE_WEIGHT = True
DIAGONAL = 'diagonal_0'
NOISE_ITERATIVE = True

np.random.seed(42)


def biased_random_patterns(size, num_patterns=1):
    patterns = np.sign(0.5 + np.random.uniform(-1, 1, size * num_patterns))
    return np.where(patterns >= 0, 1, -1).reshape(-1, size)


def random_patterns(size, num_patterns=1):
    return np.array(np.random.choice([-1, 1], size=size * num_patterns)).reshape(-1, size)

def main():
    if PATTERNS:
        num_units = 50
        num_patterns = 150
        if BIASED_PATTERNS:
            patterns = biased_random_patterns(num_units, num_patterns)  # Generating biased random patterns
        else:
            patterns = random_patterns(num_units, num_patterns)  # Generating random patterns
        if ITERATIVE_WEIGHT:
            counter = np.zeros(num_patterns)
            for i in tqdm(range(num_patterns)):
                w = weights(patterns[:i + 1, :].reshape(i + 1, num_units), diagonal=DIAGONAL)
                for j in range(i + 1):
                    if NOISE_ITERATIVE:
                        counter_aux, x_current = noised_images([0.1], patterns, j, counter, w, return_data=True, iterative_patterns=NOISE_ITERATIVE)
                    else: 
                        x_current = recall(patterns[j:j+1, :], w, update_type="synchronous",
                                   convergence_type='energy')
                    counter[i] += iterative_patterns_accuracy(patterns[:i+1, :], x_current)
            plt.figure()
            plt.plot(counter)
            plt.show()
        else:
            w = weights(patterns, diagonal = DIAGONAL)
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, patterns, 0, counter, w, noised_iterations=ITERATIONS)
            plt.figure()
            # sns.set_style('darkgrid')
            counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
            counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
            plt.show()

    else:
        # Load data
        data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
        for pat in range(3, 8):
            print("Running network for image", pat)
            w = weights(data[:pat, :], diagonal=DIAGONAL)
            image = 0
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, data, image, counter, w)
            plt.figure()
            # sns.set_style('darkgrid')
            counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
            counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
            plt.show()
            input("Press enter to continue...")


if __name__ == "__main__":
    main()
