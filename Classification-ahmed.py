# Logistic Regression Classification
# J(w,b) = 1/m* summation* loss function
# Loss =  - yi* log(fwb) - (1-yi) * log(1-fwb)
# fwb = 1/ 1 + e^(-mx+c)
# Regularization -> +lambda/m wj to the update function



import numpy as np
import math
#Values
m = np.array([0.7, 1]) 
c = 10
y_t = [0, 0, 1, 0, 0, 1]
x_t = np.array([[100, 1], [0.75, 30], [240, 50],[1, 3],[0.5, 5],[140,50]])
learn = 0.01
hx = np.zeros(len(y_t))
trainsteps = int(input("What should the number of steps be?: "))
lambdas = 0.1

for i in range(len(x_t)):
    hx[i] = 1/ (1 + pow(math.e, -np.dot(m, x_t[i]) - c))


def sumofdiff1(n_pred,n_test, n, lam):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])*x_t[i][n]
    regular = (lam/len(n_pred)) * m[n]
    return sum/len(n_pred) + regular

def sumofdiff0(n_pred,n_test):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])
    return sum/(len(n_pred))

for step in range(trainsteps):
    for i in range(len(m)):
        m[i] -= learn * sumofdiff1(hx, y_t, i, lambdas)
    c -= learn * sumofdiff0(hx, y_t)
    

    for i in range(len(x_t)):
        hx[i] = 1/ (1 + pow(math.e, -np.dot(m, x_t[i]) - c))


print(f"This is the predicted y: {hx}")
