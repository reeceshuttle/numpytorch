import numpy as np
import abc

import numpytorch as npt

class Layer:
    @abc.abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Overrride me!")

    @abc.abstractmethod
    def backward(self, dLdZ: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Overrride me!")
        

class Linear(Layer):
    def __init__(self, nodes_in: int, nodes_out: int, last_linear: bool=False, biases: bool=True):
        self.last_linear = last_linear # used to avoid matrix op, means last linear layer in backprop
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.biases = biases
        # self.lr = lr
        def weight_init(): return np.random.random()-0.5
        self.weight_init_method = weight_init

        self.W = np.array([[self.weight_init_method() for _ in range(nodes_out)] for _ in range(nodes_in)])
        if biases: 
            self.bias = np.array([[self.weight_init_method()] for _ in range(nodes_out)])

        # maybe do this:
        # self.optimizer = optimizer
        # and use it to access things like lr as they change.
        
    def forward(self, X):
        self.forward_X = X
        Y = self.W.T@X
        if self.biases:
            Y += self.bias
        return Y
        
    def backward(self, dLdY, lr):
        dLdW = self.forward_X*(dLdY.T)
        # update:
        self.W -= lr*dLdW
        self.bias -= lr*dLdY

        if not self.last_linear: # avoiding unnecesary calculation
            dLdZ = self.W@dLdY
            return dLdZ
        else: 
            return "exit backprop"


class Identity(Layer):
    def forward(self, Y):
        self.Y = Y
        return Y

    def backward(self, dLdZ, lr):
        return dLdZ
    
class ReLU(Layer):
    def forward(self, Y):
        self.Y = Y
        Z = Y * (Y>0)
        return Z

    def backward(self, dLdZ, lr):
        diagonal_vals = 1*(self.Y>0)
        dZdY = np.diag(np.squeeze(diagonal_vals))
        dLdY = dZdY@dLdZ
        return dLdY

class Sigmoid(Layer):
    def forward(self, Y):
        self.Y = Y
        dLdY = 1/(1+np.exp(-Y))
        return dLdY

    def backward(self, dLdZ, lr):
        diagonal_vals = self.forward(self.Y)*(1-self.forward(self.Y)) # store this for cheaper computation? maybe even forward pass?
        dZdY = np.diag(np.squeeze(diagonal_vals))
        dLdY = dZdY@dLdZ
        return dLdY

class Softmax(Layer):
    def forward(self, Y):
        self.Y = Y
        num = np.exp(Y)
        denom = np.sum(num)
        return num/denom

    def backward(self, dLdZ, lr):
        dZdY = np.diag(np.squeeze(self.forward(self.Y)))-(self.forward(self.Y)@(self.forward(self.Y).T))
        dLdY = dZdY@dLdZ
        return dLdY
