from Hopfield_Network import *
import pandas as pd

'extra data'
# x = np.array(pd.read_csv("pict.dat", sep=',', header=None))

x = generate_data("original")
x_distorted = generate_data("distorted")
x_super_distorted = generate_data("super_distorted")

w = weights(x)
x_updated = recall(x, w, update_type="synchronous")
x_updated_distorted = recall(x_distorted, w, update_type="synchronous")
x_updated_super_distorted = recall(x_super_distorted, w, update_type="synchronous")


print("\nOriginal data:\n", x)
print("x updated distorted:\n", x_updated_distorted)
print("x updated super distorted:\n", x_updated_super_distorted)

attractors = find_attractors(x, x_updated_distorted)
print("The attractors are: ")
print(attractors)

# the distorted converge with 2 iterations and get the right result
# the super distorted converge with 2 iterations, however it does not get the right result
# WHAT IS AN ATTRACTOR?
