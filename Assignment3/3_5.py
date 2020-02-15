from Hopfield_Network import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 50
PATTERNS = True
ITERATIVE_WEIGHT = True
DIAGONAL = ['diagonal_0', 'diagonal_not_0']
DIAGONAL_LABELS = ['Diagonal 0', 'Diagonal not 0']
NOISE_ITERATIVE = True
BIASED_PATTERNS = False

np.random.seed(42)


def biased_random_patterns(size, num_patterns=1):
    patterns = np.sign(0.5 + np.random.uniform(-1, 1, size * num_patterns))
    return np.where(patterns >= 0, 1, -1).reshape(-1, size)


def random_patterns(size, num_patterns=1):
    return np.array(np.random.choice([-1, 1], size=size * num_patterns)).reshape(-1, size)


def main():
    if PATTERNS:
        num_units = 100
        num_patterns = 60
        if BIASED_PATTERNS:
            patterns = biased_random_patterns(num_units, num_patterns)  # Generating biased random patterns
        else:
            patterns = random_patterns(num_units, num_patterns)  # Generating random patterns
        if ITERATIVE_WEIGHT:
            for label, diagonal in enumerate(DIAGONAL):
                counter = np.zeros(num_patterns)
                for i in tqdm(range(num_patterns)):
                    w = weights(patterns[:i + 1, :].reshape(i + 1, num_units), diagonal=diagonal)
                    for j in range(i + 1):
                        if NOISE_ITERATIVE:
                            counter_aux, x_current = noised_images([0.4], patterns, j, counter, w, return_data=True, iterative_patterns=NOISE_ITERATIVE)
                        else:
                            x_current = recall(patterns[j:j + 1, :], w, update_type="synchronous",
                                                convergence_type='energy')
                        counter[i] += iterative_patterns_accuracy(patterns[:i + 1, :], x_current)
                plt.plot(np.linspace(start=1, stop=num_patterns, num=len(counter)), counter,label=DIAGONAL_LABELS[label])
            plt.xlabel('Training patterns')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
        else:
            w = weights(patterns, diagonal='diagonal_not_0')
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, patterns, 0, counter, w, noised_iterations=ITERATIONS)
            plt.plot(PERCENTAGES, counter)
            plt.xlabel('Percentage')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    else:
        # Load data
        data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
        for pat in range(4, 8):
            print("Running network for image", pat)
            w = weights(data[:pat, :], diagonal='diagonal_not_0')
            image = 0
            counter = np.zeros(len(PERCENTAGES))
            counter = noised_images(PERCENTAGES, data, image, counter, w, noised_iterations=ITERATIONS)
            plt.figure()
            counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
            counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
            plt.show()
            input("Press enter to continue...")


if __name__ == "__main__":
    main()
