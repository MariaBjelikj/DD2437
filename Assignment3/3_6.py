from Hopfield_Network import *
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


SPARSE_PATTERN = True
NOISE_ITERATIVE = True
THETA = 0

def random_patterns(size, num_patterns=1):
    return np.array(np.random.choice([-1, 1], size=size * num_patterns)).reshape(-1, size)

num_units = 10
num_patterns = 10
patterns = random_patterns(num_units, num_patterns)  # Generating biased random patterns

counter = np.zeros(num_patterns)
for i in tqdm(range(num_patterns)):
    w = weights(patterns[:i + 1, :].reshape(i + 1, num_units), sparse_pattern=SPARSE_PATTERN)
    for j in range(i + 1):
        counter_aux, x_current = noised_images([0.1], patterns, j, counter, w, return_data=True, iterative_patterns=NOISE_ITERATIVE,theta = THETA)
        counter[i] += iterative_patterns_accuracy(patterns[:i+1, :], x_current)
plt.figure()
plt.plot(counter)
plt.show()
