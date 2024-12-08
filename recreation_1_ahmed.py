#Linear Regression and gradient descent from scratch

m = 10
c = 10
y_t = [60, 70, 80, 90, 100, 110,120,130,140,150]
x_t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
learn = 0.01
hx = []
trainsteps = int(input("What should the number of steps be?: "))

for i in range(len(x_t)):
    hx.append(m*x_t[i] + c)


def sumofdiff1(n_pred,n_test):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])*x_t[i]
    return sum/(100*len(n_pred))

def sumofdiff0(n_pred,n_test):
    sum = 0
    for i in range(len(n_pred)):
        sum = sum + (n_pred[i] - n_test[i])
    return sum/(100*len(n_pred))

for step in range(trainsteps):
    m -= learn * sumofdiff1(hx, y_t)
    c -= learn * sumofdiff0(hx, y_t)

    # Update predictions
    for i in range(len(x_t)):
        hx[i] = m * x_t[i] + c


print(f"This is the predicted y: {hx}")
