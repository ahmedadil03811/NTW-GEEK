# Activation For Neural Network: {a_j}^[L]  = g({w_j}^[L] . a^[L-1] + {b_j}^[L])

# layer_1 = DENSE(units = 25, activation='sigmoid')
# x = numpy.array([123,10])
# a1 = layer_1(x)

import numpy as np
import math

W1 = np.array( [[1,  2 , 3 ], [-0.1,  2, 5]] )
b1 = np.array( [-6, 5,  0.1] )
W2 = np.array( [[5], [-2], [-6]] )
b2 = np.array([4])

x_t = np.array([[100, 17], [200, 50], [300, 5]])

y_t = np.array([1, 0, 1])

learn = 0.1

def sig(hx):
    ax_out = 1/ (1 + pow(math.e, -1*hx))
    return ax_out
    
def Dense(a_in, W, b, g):
    ia = W.shape[1]
    a_out = np.zeros(ia)
    for i in range(ia):
        w = W[:,i]
        l = np.dot(w, a_in) + b
        a_out[i] = g(l)
    return a_out

def Sequential_1(x, W1, b1, W2, b2):
    a1 = Dense(x,  W1, b1, sig)
    a2 = Dense(a1, W2, b2, sig)
    return a1, a2

def sumofdiff1(n_pred,n_test, n):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])*x_t[i][n]
    return sum/len(n_pred)

def sumofdiff0(n_pred,n_test):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])
    return sum/(len(n_pred))



def train(W1, b1, W2, b2, x_t, y_t, learn, steps):
    ## for week 2




