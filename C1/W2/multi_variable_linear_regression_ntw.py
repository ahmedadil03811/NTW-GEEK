import numpy as np
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
m = X_train.shape[0]
a = 0.0000001
steps = 1000000

# Create Parameters w and b
w = np.array([1,1,1,1])
b = 1
#Calculating Derivatives 
def d_w(w,b):
    sum_w = np.zeros(len(w))
    for j in range(len(w)):
        sum = 0
        for i in range(X_train.shape[0]):
            sum += ((np.dot(w,X_train[i]) + b - y_train[i])*X_train[i][j])
        sum_w[j] = sum
    sum_w = sum_w/m
    return sum_w
def d_b(w,b):
    sum = 0
    for i in range(X_train.shape[0]):
        sum += (np.dot(w,X_train[i]) + b - y_train[i])
    sum_b = sum/m
    return sum_b
################### wb() ####################
def wb(w,b):
    for step in range(steps):
        temp_w = w - a * d_w(w,b)
        temp_b = b - a * d_b(w,b)
        w = temp_w
        b = temp_b
    return w,b
################## main() ####################
def main():
    global w,b
    x_input = np.zeros(len(w))
    w,b = wb(w,b)
    for i in range(len(w)):
        x_input[i] = int(input(f"Input{i+1}: "))
    output = np.dot(w,x_input) + b
    print(output)

if __name__ == "__main__":
    main()