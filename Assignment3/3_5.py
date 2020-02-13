from Hopfield_Network import *
import numpy as np
import seaborn as sns

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 100


def random_patterns(size):
    data = np.random.choice([-1,1],size=size)
    return np.array(data)


# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

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



    
