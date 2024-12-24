import numpy as np
import math

W1 = np.array( [[1,2,3,4], [3,4,5,6]], dtype=np.float64) 
b1 = np.array( [-6, 5,  0.1, 2], dtype=np.float64)
W2 = np.array( [[1.0,-1, -0.01, 0.001]], dtype=np.float64)
b2 = np.array([4], dtype=np.float64)
x_t = np.array([[3, 7], [20, 50], [3, 5], [35, 52]], dtype=np.float64)
y_t = np.array([1, 0, 1, 0], dtype=np.float64)
m = 2


def sig(hx):
    ax_out = 1/(1+np.exp(-hx))
    return ax_out

def relu(hx):
    ax_out = np.maximum(0, hx)
    return ax_out

def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)
    

def sig_deriv(Z):
    return z*(1-z)

def forward_prop(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2.T) + b2
    a2 = sig(z2)
    print(a2)
    return z1, a1, z2, a2


#c = w1*x -> z1 = c+b1 -> a1 = relu(z1) -> c = w2*a1 -> z2 = c+b2 -> a2 = sig(z2) -> loss = ((a2-Y)/m)*a1

def backward_prop(z1,a1,z2,a2,W1,W2,X,Y):
    m = X.shape[0]
    # Compute error at the output layer
    dz2 = a2 - Y.reshape(-1, 1)
    dW2 = (1/m) * np.dot(dz2.T, a1)
    db2 = (1/m) * np.sum(dz2, axis=0)
    
    # Compute error at the hidden layer
    dz1 = np.dot(dz2, W2) * ReLU_deriv(z1)
    dW1 = (1/m) * np.dot(dz1.T, X).T
    db1 = (1/m) * np.sum(dz1, axis=0)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learn):
    W1 = W1 - learn * dW1
    b1 = b1 - learn * db1    
    W2 = W2 - learn * dW2  
    b2 = b2 - learn * db2   
    return W1, b1, W2, b2



def train(W1, b1, W2, b2, X, Y, alpha, iterations):
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X,W1,b1,W2,b2)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    return W1, b1, W2, b2, 

train(W1,b1,W2,b2,x_t,y_t,0.001, 1000)




