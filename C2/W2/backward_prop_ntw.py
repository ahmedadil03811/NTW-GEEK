import numpy as np
import math as math
X_train = np.array([[10, 1.5], [20, 1], [30, 0.5], [40, 0.5], [50, 2], [60, 2.5]])
Y_train = np.array([0, 0, 0, 1, 1, 1])
W1 = np.array([[1,1],[1,1]])
B1 = np.array([[1],[1]])
W2 = np.array([[0.008, 0.009]])
B2 = np.array([[1]])
alpha = 0.001
steps = 1000

def g(z,func_name):
    if func_name == "ReLu":
        return max(0,np.take(z,0))
    if func_name == "Sigmoid":
        z = np.take(z,0)
        return (1/(1+pow(math.e,-z)))
def diff_g(z,func_name):
    if func_name == "ReLu":
        if z >= 0:
            x = 1
        else:
            x = 0
        return x
    elif func_name == "Sigmoid":
        return (g(z,func_name)*(1-(g(z,func_name))))
    
# Forward Propagation
W = [W1, W2]
B = [B1,B2]
G = ["ReLu","Sigmoid"]
def forward_prop(x,weights,biases):
    layers = len(weights)
    units_l = []
    layer_input = [x]
    layer_output = []
    dzw = []
    daz = []
    dza_l = []
    for layer in range(layers):
        units = len(weights[layer])
        node_output = []
        dzw_u = []
        daz_u = []
        for unit in range(units):
            w = (weights[layer][unit])
            b = (biases[layer][unit])
            if len(layer_input[layer]) == 1:
                return units,layers,layer_output,dzw,daz,dza_l
            if isinstance(w[0], np.ndarray):
                w_t = []
                for i in range(len(w)):
                    w_t.append(np.take(w,i))
                w = w_t
            z = np.dot(w,np.array(layer_input[layer])) + b
            node_output.append(g(z,G[layer]))
            dzw_uw = []
            for weight in range(len(w)):
                dzw_uw.append(layer_input[layer][weight])
            dzw_u.append(dzw_uw)
            daz_u.append(diff_g(z,G[layer]))
        dzw.append(dzw_u)
        daz.append(daz_u)
        dza_l.append(node_output)
        layer_output.append(node_output)
        layer_input.append(node_output)
        units_l.append(units)
    return units_l,layers,layer_output,dzw,daz,dza_l

# Backward Propagation
def backward_prop(W,B,X,Y):
    units_l,layers,l_o,dzw,daz,dza = forward_prop(X,W,B)
    ca = l_o[-1][-1]
    dca = 2 * ( ca - Y)
    weights_l = []
    biases_l = []
    for layer in range(layers):
        weights_u = []
        biases_u =[]
        units = units_l[layer]
        for unit in range(units):
            weights_w = []
            daz_i = daz[layer][unit]
            dzai = dza[layer][unit]
            if layer == (layers-1):
                daz_o = daz[layer][0]
            else: 
                daz_o = 1
            D = np.float64(daz_i) * np.float64(dzai) * np.float64(daz_o) * np.float64(dca)
            length = len(W[layer][unit])
            for weight in range(length):
                #print(dzw)
                dzwi = dzw[layer][unit][weight]
                differential_w = dzwi * D
                tempw = W[layer][unit][weight] - alpha * differential_w
                weights_w.append(tempw)
            differential_b = 1 * D
            tempb = B[layer][unit] - alpha * differential_b
            weights_u.append(weights_w)
            biases_u.append(tempb)
        biases_l.append(biases_u)
        weights_l.append(weights_u)
    return weights_l,biases_l     

def main(X,Y,steps):
    global W,B
    for step in range(steps):
        for i in range(len(Y)):
            Q,E = backward_prop(W,B,X[i],Y[i])
            W,B = Q,E
    return Q,E # Updated Parameters

# Test
def test():
    N,M = main(X_train,Y_train,steps)
    units_l,layers,l_o,dzw,daz,dza = forward_prop([10, 1.5],N,M)
    answer = l_o[-1][-1]
    print(f"The probability is {answer}")

test()

