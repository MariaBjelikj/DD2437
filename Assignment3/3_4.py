from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np

# PERCENTAGES = np.linspace(0.1, 0.9, num=9)
PERCENTAGES = [0.01]

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

x = data[:3, :].copy()
w = weights(x)

for i in PERCENTAGES:
    for j in range(0, data.shape[0]):
        indexes = shuffle(np.arange(0, data.shape[1]))
        for k in range(int(i * data.shape[1])):
            data[j, indexes[k]] = -data[j, indexes[k]]
    # Once the determined percentage has been modified, now it's time to run the recall
    # Random asynchronous (uncomment code for 3.2 in Hopfield_Network)
    # Learn the first 3 patterns
    #for i in range(11):
    #    display(data[i])
    display(data[0])
    x_current = recall(data[0:1, :].copy(), w, update_type="synchronous", convergence_type='energy')
    display(x_current)
