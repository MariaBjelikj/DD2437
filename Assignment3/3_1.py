import numpy as np
from Hopfield_Network import *

x = generate_data("original")
x_distorted =  generate_data("distorted")
x_super_distorted =  generate_data("super_distorted")

w = weights(x)
x_updated = update_syncroniously(x, w)
x_updated_distorted = update_syncroniously(x_distorted, w)
x_updated_super_distorted = update_syncroniously(x_super_distorted, w)
