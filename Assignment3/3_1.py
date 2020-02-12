from Hopfield_Network import *
import pandas as pd
import itertools

def generate_data(d_type):
    # For task 3.1
    if d_type == "original":
        x1d = [-1, -1, 1, -1, 1, -1, -1, 1]
        x2d = [-1, -1, -1, -1, -1, 1, -1, -1]
        x3d = [-1, 1, 1, -1, -1, 1, -1, 1]
        return np.vstack([x1d, x2d, x3d])

    elif d_type == "distorted":
        x1d = [1, -1, 1, -1, 1, -1, -1, 1]  # one bit error
        x2d = [1, 1, -1, -1, -1, 1, -1, -1]  # two bit error
        x3d = [1, 1, 1, -1, 1, 1, -1, 1]  # two bit error
        return np.vstack([x1d, x2d, x3d])

    elif d_type == "super_distorted":
        # more then half the data is distroted
        x1d = [1, 1, -1, -1, -1, 1, -1, 1]
        x2d = [1, 1, 1, -1, -1, -1, 1, -1]
        x3d = [1, -1, -1, 1, -1, 1, -1, -1]
        return np.vstack([x1d, x2d, x3d])

    elif d_type == 'all_binary_comb':
        permutation = list(itertools.product([-1,1],repeat=8))
        return np.array(permutation)




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


attractor_data = generate_data("all_binary_comb")

attractors = find_attractors(attractor_data, w, update_type="synchronous")
print("The attractors are: ")
print(attractors)
print(attractors.shape[0])

# the distorted converge with 2 iterations and get the right result
# the super distorted converge with 2 iterations, however it does not get the right result
# WHAT IS AN ATTRACTOR?
