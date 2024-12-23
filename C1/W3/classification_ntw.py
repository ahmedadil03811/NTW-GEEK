import numpy as np
X_train = np.array([[10, 1.5], [20,1], [30, 0.5], [40, 0.5], [50, 2], [60, 2.5]])
Y_train = np.array([0, 0, 0, 1, 1, 1])
w = [1,1]
b = 1
m = X_train.shape[0]
alpha = 0.001 # most optimal learning rate
steps = 10000 # most optimal step count
######################## calculate derivatives ######################
def d_w(w,X_train,m):
    temp_sum_w = np.zeros(len(w))
    for i in range(len(w)):
        sum = 0
        for j in range(m):
            f_wb = 1/(1+2.7**(-(np.dot(w,X_train[j]) + b))) 
            sum += (f_wb - Y_train[j])*X_train[j][i]   #(f_wb(X_i) - Y_i)(X_i)
        sum_w = sum/m
        temp_sum_w[i] = sum_w
    return temp_sum_w

def d_b(w,X_train,m):
    sum = 0
    for j in range(m):
        f_wb = 1/(1+2.7**(-(np.dot(w,X_train[j]) + b))) 
        sum += f_wb - Y_train[j]
    sum = sum/m
    return sum
######################### main ############################
def main(w,b,X_train,m):
    for step in range(steps):
        temp_w = w - alpha * d_w(w,X_train,m)
        temp_b = b - alpha * d_b(w,X_train,m)
        w = temp_w
        b = temp_b
    return w,b
    
#w,b = main(w,b,X_train,m)
#print(w,b)
# Optional
def tumor(w,b,X_train,m):
    w,b = main(w,b,X_train,m)
    age = int(input("Age: "))
    size = float(input("Tumor Size: "))
    P = 1/(1+2.7**(-(np.dot(w,[age,size] + b))))
    print(f"Probability of Tumor being malignant: {P}")

tumor(w,b,X_train,m)