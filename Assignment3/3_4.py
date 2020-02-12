from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np

PERCENTAGES = [0.1, 0.3, 0.5, 0.7, 0.9]
#PERCENTAGES = np.arange(0.1,0.9,9)

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

x = data[:3, :].copy()
w = weights(x)

image = 0

for i in PERCENTAGES:
    for j in range(0, data.shape[0]):
        indexes = shuffle(np.arange(0, data.shape[1]))
        for k in range(int(i * data.shape[1])):
            data[j, indexes[k]] = -data[j, indexes[k]]

    noised_filename = 'images/noised_error{}'.format(i)
    noised_filename = noised_filename.replace('.', ',', 1)
    noised = display(data[image], title = "Original noised imaged with error = {}".format(i), save=True, filename=noised_filename)
    
    recovered_filename = 'images/recovered_error{}'.format(i)
    recovered_filename = recovered_filename.replace('.', ',', 1)
    x_current = recall(data[image:(image+1), :].copy(), w, update_type="synchronous", convergence_type='energy')
    recovered = display(x_current, title = "Recovered noised imaged with error = {}".format(i), save=True, filename=recovered_filename)


    
