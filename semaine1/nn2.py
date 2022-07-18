from hashlib import sha512
import numpy as np
import math
import math 
import import_data


layer_sizes = [784,40,10]

def sigm(x):
    return 1/(1 + math.e **(-x))
def d_sigm(x):
    return (1 + math.e**(-x))**-2 * math.e**(-x)

## initialize
m = 10
data = import_data.load_data()
y = data[0]
x = data[1]
w1 = np.random.rand(layer_sizes[1],layer_sizes[0])
b1 = np.random.rand(layer_sizes[1],1)
w2 = np.random.rand(layer_sizes[2], layer_sizes[1])
b2 = np.random.rand(layer_sizes[2],1)


def data_prediction(w1,b1,w2,b2,x):
    print("w1", w1[:,:10])
    z1 = np.matmul(w1,x) + b1
    print("z1",z1[:,:10])
    a1 = sigm(z1)
    print("a1", a1[:,:10])
    z2 = np.matmul(w2, a1) + b2
    print("w2", w2[:,:10])
    print("b2",b2)
    print("z2",z2[:,:10])
    a2 = sigm(z2)
    return a2



print(y.shape)
pointer = 0
learning_rate = 0.055
if __name__ == "__main__":
    pass
    ## learning
    for elements in range(1000):

        sample_size = 50
        print(elements)
        ### front propagation
        z1 = np.matmul(w1,x[:,pointer:pointer + sample_size]) + b1
        a1 = sigm(z1)
        z2 = np.matmul(w2, a1) + b2
        a2 = sigm(z2)

        ## backpropagation
        dz2 = a2 - y[:, pointer:pointer+sample_size]
        dw2 = (1/m) * np.matmul(dz2,np.transpose(a1))
        db2 =  (1/m) * np.sum(dz2,axis = 1)
        dz1 = np.matmul(np.transpose(w2), dz2) * d_sigm(z1)
        dw1 = (1/m) * np.matmul(dz1,np.transpose(x[:,pointer:pointer + sample_size]))
        db1 =  (1/m) * np.sum(dz1,axis = 1)
        pointer += sample_size

        ## gradient descent
        w1 = w1 - learning_rate*dw1
        w2 = w2 - learning_rate*dw2
        b1 = b1 - learning_rate*np.reshape(b1, (layer_sizes[1],1))
        b2 = b2 - learning_rate*np.reshape(b2, (layer_sizes[2],1))
        cost_matrix_j = -((np.log(a2) * y[:,pointer-sample_size:pointer]) + (1-y[:,pointer-sample_size:pointer]) * np.log(1-a2))
        cost_average = np.average(cost_matrix_j, axis = 1)
        print(cost_average)

        if elements % (500/sample_size) == 0:
            pointer = 0

    print("x", x[:,1])
    y_hat = data_prediction(w1,b1,w2,b2,x)
    print(y_hat[:,:10])
    print(y[:,:10])


