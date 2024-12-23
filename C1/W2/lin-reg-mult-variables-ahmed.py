import numpy as np

m = np.array([1,2,3,7]) 
c = 10
y_t = [60, 70, 80, 90]
x_t = np.array([[1,2,3,4], [2, 3, 4, 5], [3, 4, 5, 6],[4,5,6,7]])
learn = 1
hx = np.zeros(len(y_t))
trainsteps = int(input("What should the number of steps be?: "))

for i in range(len(x_t)):
    hx[i] = np.dot(m, x_t[i]) + c


def sumofdiff1(n_pred,n_test, n):
    sum = 0
    print(hx)
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])*x_t[i][n]
    return sum/1000*(len(n_pred))

def sumofdiff0(n_pred,n_test):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])
    return sum/1000*(len(n_pred))

for step in range(trainsteps):
    for i in range(len(m)):
        m[i] -= learn * sumofdiff1(hx, y_t, i)
    c -= learn * sumofdiff0(hx, y_t)
    
    for i in range(len(x_t)):
        hx[i] = np.dot(m, x_t[i]) + c


print(f"This is the predicted y: {hx}")

print(f"Values of m: {m} and Value of c: {c}")


asl = np.array(input("What 4 values do you wish to predict for (comma-separated): ").split(','), dtype=float)

value = np.dot(m, asl) + c
print(f"Predicted value is: {value}")

