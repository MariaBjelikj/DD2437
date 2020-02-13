from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import pandas as pd

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 100

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

w = weights(data[:3, :])
image = 0
counter = np.zeros(len(PERCENTAGES))
counter = noised_images(PERCENTAGES, data, image, counter, w)
plt.figure()
# sns.set_style('darkgrid')
counter_plot = sns.lineplot(x = PERCENTAGES, y=counter)
counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
plt.show()

# noised_filename = 'images/image{}_error{}_noised'.format(image, i)
# noised_filename = noised_filename.replace('.', ',', 1)
# noised = display(noised_data[image], title = "Original noised imaged with error = {} for image {}".format(i,image), save=False, filename=noised_filename)

# recovered_filename = 'images/image{}_error{}_recovered'.format(image, i)
# recovered_filename = recovered_filename.replace('.', ',', 1)
# x_current = recall(noised_data[image:(image+1), :].copy(), w, update_type="synchronous", convergence_type='energy')
# recovered = display(x_current, title = "Recovered noised imaged with error = {} for image {}".format(i, image), save=False, filename=recovered_filename)


    
