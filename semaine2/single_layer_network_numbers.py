#!/usr/bin/env python
# coding: utf-8

# In[147]:


import csv
import numpy as n
"""import matplotlib.pyplot as plt
import matplotlib.image as mpimg"""

def load_data():

    keeper = 0
    with open("/Users/billxue/Documents/neural_network/mnist_train.csv", encoding = "utf-8") as f:
        reader = csv.reader(f)

        y = n.zeros((10,1))
        x = n.zeros((784, 1))
        for elements in reader:
            a = n.zeros((10,1))
            a[int(elements[0]), 0] = 1
            y = n.append(y, a, axis = 1)
            x = n.append(x, n.array([[int(x)/255] for x in elements[1:]]), axis = 1)
            keeper+=1
            if keeper % 1000 == 0:
                print(print(x.shape))
            if keeper == 15000:
                break
    """data= [( int(x[0]), n.array([[int(k)/255] for k in x[1:]]) ) for x in reader]
        for elements in range(10):
            print("\n\n")
            print(data[elements])"""

    return (y[:,1:10000], x[:,1:10000], y[:,10000: 15000], x[:,10000:15000])


# In[148]:


damn = load_data()


# In[150]:


import math
def sigm(x):
    return 1/(1 + math.e **(-x))
Y = damn[0]
X = damn[1]


# In[151]:


def initiate_para(X,Y, hiddenl_size):
    W1 = n.random.rand(hiddenl_size,X.shape[0]) * 0.01
    W2 = n.random.rand(Y.shape[0], hiddenl_size) * 0.01
    b1 = n.zeros((hiddenl_size, 1))
    b2 = n.zeros((Y.shape[0], 1))
    return {"W1": W1, "W2": W2, "b1": b1, "b2": b2}


# In[152]:


param = initiate_para(X,Y,26)


# In[153]:


def front_p(param, X):
    Z1 = n.matmul(param["W1"], X) + param["b1"]
    A1 = n.tanh(Z1)
    Z2 = n.matmul(param["W2"], A1) + param["b2"]
    A2 = sigm(Z2)
    
    return {"Z1" : Z1,"A1" : A1,"Z2" : Z2, "A2" : A2}


# In[154]:


cache = front_p(param, X)


# In[155]:


cache["Z2"]


# In[156]:


def cost_computation(A2,Y):
    m = A2.shape[1]
    ny = Y.shape[0]
    cost_by_digit = (1/m) * n.sum(-(Y * n.log(A2) + (1-Y)*n.log(1-A2)),axis = 1)
    cost_total = (1/ny) * n.sum(cost_by_digit, axis = 0)
    return (cost_by_digit, cost_total)


# In[157]:


hell = cost_computation(cache["A2"], Y)


# In[ ]:





# In[158]:


def back_p(param, cache, Y, X):
    m = Y.shape[1]
    W2 = param["W2"]
    b2 = param["b2"]
    W1 = param["W1"]
    b1 = param["b1"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    # use memory as an organisational tool 
    
    dZ2 = A2 - Y
    dW2 = (1/m) * n.matmul(dZ2,A1.T)
    db2 = (1/m) * n.sum(dZ2, axis = 1, keepdims = True)
    
    dZ1 = n.matmul(W2.T,dZ2) * (1-A1**2)
    dW1 = (1/m) * n.matmul(dZ1,X.T)
    db1 = (1/m) * n.sum(dZ1, axis = 1, keepdims = True)
    
    return {"dW2" : dW2, "db2": db2, "dW1" : dW1, "db1" : db1}


# In[159]:


grads = back_p(param, cache, Y, X)


# In[160]:


grads


# In[161]:


grads["dW1"].shape


# In[162]:


def descent(params, grads, learning_rate = 0.5):
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = params["W1"] - learning_rate * dW1
    b1 = params["b1"] - learning_rate * db1
    W2 = params["W2"] - learning_rate * dW2
    b2 = params["b2"] - learning_rate * db2
    
    return {"W1": W1 , "W2": W2, "b1": b1, "b2": b2}


# In[163]:


new = descent(param, grads)


# In[206]:


def neural_network(X,Y,hiddenl_size,iterations = 20000, batch_size = 1000):
    param = initiate_para(X,Y, hiddenl_size)
    m = Y.shape[1]
    cursor = 0
    for elements in range(iterations):
        
        batch_x = X[:,cursor:cursor + batch_size]
        batch_y = Y[:,cursor:cursor + batch_size]
        cache = front_p(param, batch_x)
        
        if elements % (iterations/10) == 0:
            print(cost_computation(cache["A2"],batch_y)[1])
            
        grads = back_p(param, cache, batch_y,batch_x)
        param = descent(param, grads, learning_rate = 0.25)
        
        if cursor + batch_size >= m:
            cursor = 0
        else:
            cursor += batch_size
            
            
    return param
        
        
        
        
        
        


# In[219]:


params = neural_network(X,Y,60)


# In[215]:


X_test = damn[3]
Y_test = damn[2]


# In[ ]:





# In[220]:


def prediction (X_test, Y_test, params):
    predictions = front_p(params, X_test)["A2"]
    print(predictions[:,:2])
    predictions = predictions.argmax(axis = 0)
    
    return predictions
    


# In[221]:


def comparison(prediction, Y_test):
    predictions = prediction (X_test, Y_test, params)
    y_test_m = Y_test.argmax(axis =0)

    outlier = predictions - y_test_m
    count_diff = n.count_nonzero(outlier)
    
    accuracy = (Y_test.shape[1] - count_diff)/Y_test.shape[1]
    return accuracy


# In[222]:


comparison(prediction, Y_test)


# In[144]:


X_test = damn[1]
Y_test = damn[2]


# In[149]:


prediciton = np.array([1,2,3])
data = np.array([1,0,3])


# In[ ]:




