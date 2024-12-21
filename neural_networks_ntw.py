import numpy as np
import math as math
X_train = np.array([10, 1.5]) #, [20, 1], [30, 0.5], [40, 0.5], [50, 2], [60, 2.5]])
# Y_train = np.array([0, 0, 0, 1, 1, 1])
W1_ = np.array([[1,1],
              [2,2],
              [3,3]])
# print(len(W1_))
B1_ = np.array([[1],
               [2],
               [3]])
W2_ = np.array([0.008,1.01,0.009])
B2_ = np.array([1])
def sigmoid(w,x,b):
    return (1/(1+pow(math.e,(-(np.dot(w,x)+b)))))

# print(sigmoid([1,2],[0,0.05],[1]))

def layer(W,A_old,B):
    l = len(W1_)
    a_temp = np.zeros(l)
    # print(a_temp)
    for i in range(l):
        a_temp[i] = sigmoid(W[i],A_old,B[i])
    return a_temp

def main():
    global W1_,W2_,X_train,B1_,B2_
    a = layer(W1_,X_train,B1_)

    b = sigmoid(W2_,a,B2_)

    print(b[0])

main()









