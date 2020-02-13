from Hopfield_Network import *
import numpy as np
import seaborn as sns

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 100
PATTERNS = True


def random_patterns(size):
    return np.array(np.random.choice([-1, 1], size=size)).reshape(-1, size)


# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)
patterns = random_patterns(1024 * 8)

if PATTERNS:
    w = weights(patterns)
    image = 0
    counter = np.zeros(len(PERCENTAGES))
    counter = noised_images(PERCENTAGES, patterns, image, counter, w, noised_iterations=100)
    plt.figure()
    # sns.set_style('darkgrid')
    counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
    counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
    plt.show()
    input("Press enter to continue...")

else:
    for pat in range(4, 8):
        print("Running network for image", pat)
        w = weights(data[:pat, :])
        image = 0
        counter = np.zeros(len(PERCENTAGES))
        counter = noised_images(PERCENTAGES, data, image, counter, w)
        plt.figure()
        # sns.set_style('darkgrid')
        counter_plot = sns.lineplot(x=PERCENTAGES, y=counter)
        counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
        plt.show()
        input("Press enter to continue...")
