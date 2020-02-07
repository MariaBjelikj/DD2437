from Hopfield_Network import *

# Load data
data = np.loadtxt('pict.dat', delimiter=",", dtype=int).reshape(-1, 1024)

# First train for first 3 images
x_train = data[:3, :].copy()
w = weights(x_train)

# Test for next 2 images
x_test = data[3:5, :].copy()
x_test_1 = data[8:10, :].copy()
prediction = update_syncroniously(x_test, w)
prediction_1 = update_asyncroniously(x_test, w)

display(prediction[0])
display(prediction_1[0])
