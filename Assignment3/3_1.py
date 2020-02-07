from Hopfield_Network import *
import pandas as pd

'extra data'
# x = np.array(pd.read_csv("pict.dat", sep=',', header=None))

x = generate_data("original")
x_distorted = generate_data("distorted")
x_super_distorted = generate_data("super_distorted")

w = weights(x)
x_updated = update_syncroniously(x, w)
x_updated_distorted = update_syncroniously(x_distorted, w)
x_updated_super_distorted = update_syncroniously(x_super_distorted, w)


print("\nOriginal data:\n", x)
print("x updated distorted:\n", x_updated_distorted)
print("x updated super distorted:\n", x_updated_super_distorted)

# the distorted converge with 2 iterations and get the right result
# the super distorted converge with 2 iterations, however it does not get the right result
# WHAT IS AN ATTRACTOR?
