from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

PERCENTAGES = np.linspace(start=0, stop=1, num=11)
ITERATIONS = 100

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

w = weights(data[:3, :])
image = 0
counter = np.zeros(len(PERCENTAGES))
counter = noised_images(PERCENTAGES, data, image, counter, w, noised_iterations=ITERATIONS)
plt.figure()
# sns.set_style('darkgrid')
counter_plot = sns.lineplot(x = PERCENTAGES, y=counter)
counter_plot.set(xlabel='Noise level', ylabel='Accuracy')
plt.show()
############ PLOT SPECIFIC PERCENTAGE
# counter = np.zeros(len(PERCENTAGES))
# counter, x_current = noised_images([0.9], data, image, counter, w, noised_iterations=ITERATIONS, return_data = True)
# display(x_current, title = "Recovered noised imaged with error 0.9")




    
