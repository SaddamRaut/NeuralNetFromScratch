# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class Layer():
    def __init__(self):
        self._input = None
        self._output = None
    
    def forword(self, input):
        raise NotImplementedError

    def backword(self, output_error, lr):
        raise NotImplementedError
        
    
class Dense(Layer):
    def __init__(self, input_size, output_size, log = False):
        
        self._weights = np.random.rand(input_size, output_size)
        self._bias  = np.random.rand(1, output_size)
        
        if(log):
            print("self._weights dimension : {} ".format(self._weights.shape))
            print("self._bias dimension : {} ".format(self._bias.shape))
            
                
    def forword(self, input):
        '''
        implementation  y = wx + b
        '''
        self._input = input
        self._output = np.dot(input, self._weights) + self._bias
        return self._output
    
    def backword(self, error, lr):
        '''
        computes dE/dW, dE/dB for a given error=dE/dY. 
        Returns input_error=dE/dX.
        '''
        
        de_by_dw = np.dot(self._input.T, error)
        de_by_dx = np.dot(error, self._weights.T)
        
        # updates the w and b
        self._weights -= lr * de_by_dw
        self._bias -= lr * error
        
        return de_by_dx
     
 
class Activation(Layer):
    
    def __init__(self):
        pass
    
    def tanh(self, x):
        return np.tanh(x);

    def tanh_prime(self, x):
        return 1-np.tanh(x)**2;
    
    def forword(self, input):
        self.input = input
        self.output = self.tanh(self.input)
        return self.output
    
    def backword(self, error, lr):
        return self.tanh_prime(self.input) * error
    

def MSE(y_true, y_pred):
    return np.mean(np.power((y_true - y_pred), 2))

def MSE_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class Net:
    def __init__(self):
        self.layers = []
        self.loss = MSE
        self.loss_prime = MSE_prime
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, inputs):
        result = []    
        print(inputs)

        for i in range(len(inputs)  ):
            data = inputs[i]
            output = inputs[i]
            
            for layer in self.layers:
                output = layer.forword(output)
            result.append(output)
        return result
    
    def fit(self, x_train, y_train, epochs, lr):
        samples = len(x_train)
        err = 0
        for i in range(epochs):
            error = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forword(output)
        
                    # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
    
                # backward propagation
                # de/dy
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backword(error, lr)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            
            
'''
Testing the network
'''

data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
data.shape

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[1]], [[0]], [[0]], [[1]]])

net = Net()
net.add(Dense(2, 3))
net.add(Activation())
net.add(Dense(3, 1))
net.add(Activation())

net.fit(x_train.copy(), y_train, epochs=1000, lr=0.1)

# test
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
out = net.predict(x_train)
print(out)

        