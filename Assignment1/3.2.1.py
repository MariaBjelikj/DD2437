import numpy as np
import matplotlib.pyplot as plt
import time
import DataGeneration as dg


mean = np.array([0,1])
sigma = 1
n_hidden = 5
x, t = dg.non_linearly_separable_data(mean, sigma)
v = np.random.rand(t.shape[0], n_hidden)
w = np.random.rand(n_hidden, x.shape[0])
dv = np.random.rand(t.shape[0], n_hidden)
dw = np.random.rand(n_hidden, x.shape[0])


def phi(x):
    phi = 2/(1+np.exp(-x)) - 1
    phi_derivative = (1+phi)*(1-phi)/2
    return phi, phi_derivative

def forward_pass(x, w, v):
    hin = np.dot(w, x)
    phi_h = phi(hin)[0]
    hout =  np.vstack((phi_h))
    
    oin = np.dot(v, hout)
    oout = phi(oin)[0]
    
    return hout, oout
    
def backward_pass(t, w, v, o, h):    
    delta_o = (o - t) * ((1 + o) * (1 - o)) * 0.5
    ok = 1
    delta_h = np.dot(np.transpose(v), delta_o) * ((1 + h) * (1 - h))*0.5
    delta_h = delta_h[range(delta_h.shape[0]),:]    #remove extra rows
    
    return delta_o, delta_h

def weight_update(x, dw, dv, alpha, delta_h, delta_o, h, w, v):
    dw = (dw * alpha) - np.dot(delta_o, np.transpose(x))*(1-alpha)
    dv = (dv * alpha) - np.dot(delta_h, np.transpose(h))*(1-alpha)
    W = w + dw*eta
    V = v + dv*eta
    return W, V


#print("This is x", x)
#print("\nThis is t", t)
print(t.shape)


#print("\nThis is phi of x", phi(x))


h, o = forward_pass(x, w, v)
delta_o, delta_h = backward_pass(t,w,v,o,h)

alpha = 0.9
eta = 0.001

print(weight_update(x, dw, dv,alpha, delta_h, delta_o,h, w, v))

