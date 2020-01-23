import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg
import Algorithms as alg

SAMPLES = 8

n_hidden = 3
x, t = dg.enconder_data()
v = np.random.normal(0,1,(t.shape[0], n_hidden))
w = np.random.normal(0,1,(n_hidden, x.shape[0]))
dv = np.zeros((t.shape[0], n_hidden))
dw = np.zeros((n_hidden, x.shape[0]))
alg.encoder(x, w, v, dw, dv, t)



