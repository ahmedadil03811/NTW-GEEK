# Linear Regression with Gradient Descent in one variable

x = [1000, 1200, 1500, 1700, 2000]
y = [80, 100, 130, 140, 180]

# Compute w
# assume w,b and a
w = 1
b = 1
a = 0.00000001
steps = 1000000
# calculate partial derivatives
def d_w(w,b,l=len(x)):
    sum_w = 0
    for i in range(l):
        sum_w += (w*x[i] + b - y[i])*x[i]
    sum_w = sum_w/l
    return sum_w
def d_b(w,b,l=len(x)):
    sum_b = 0
    for i in range(l):
        sum_b += (w*x[i] + b - y[i])
    sum_b = sum_b/l
    return sum_b
# main function
def main(w,b):
    tmp_w = 0
    tmp_b = 0
    for step in range(steps):
        if (d_w(w,b) != 0):
            tmp_w = w - a*d_w(w,b)
        if (d_w(w,b)!=0 ):
            tmp_b = b - a*d_b(w,b)
        w = tmp_w
        b = tmp_b
    return w,b


def house():
    global w,b
    w,b = main(w,b)
    user_input = int(input("Input Size: "))
    print(f"Expected Price: {(w*user_input+b)*1000} Dollars")

if __name__ == "__main__":
    house()









