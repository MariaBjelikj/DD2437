from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns

PERCENTAGES = np.linspace(start = 0, stop = 1, num = 5)
ITERATIONS = 1


def random_patterns(size):
    data = np.random.choice([-1,1],size=size)
    return np.array(data)


test = random_patterns(1024)

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

x = data[:4 , :].copy()
w = weights(x)

image = 0

counter = np.zeros(len(PERCENTAGES))
for i,perc in enumerate(PERCENTAGES):
    for _ in range(ITERATIONS):
        noised_data = np.copy(data)
        indexes = shuffle(np.arange(0, noised_data.shape[1]))
        for k in range(int(perc * noised_data.shape[1])):
            noised_data[image, indexes[k]] = -noised_data[image, indexes[k]]
        x_current = recall(noised_data[image:(image+1), :], w, update_type="synchronous", convergence_type='energy')
        #display(noised_data[image])
        #display(x_current)
        if np.all(x_current == data[image]):
            counter[i] += 1


plt.figure()
#sns.set_style('darkgrid')
counter_plot = sns.lineplot(x = PERCENTAGES, y=counter)
counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
plt.show()



    
