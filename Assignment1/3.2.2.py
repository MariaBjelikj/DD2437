import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

n_hidden = 2
x, t = dg.enconder_data()
#sigma = np.sqrt(2/x.shape[1]) Xavier and He initialization
v = np.random.normal(0,1,(t.shape[0], n_hidden))
w = np.random.normal(0,1,(n_hidden, x.shape[0]))
dv = np.zeros((t.shape[0], n_hidden))
dw = np.zeros((n_hidden, x.shape[0]))
alg.encoder(x, w, v, dw, dv, t)



