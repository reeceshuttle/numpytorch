import numpy as np
import abc

from numpytorch.activation import Activation

class Layer:
    @abc.abstractmethod
    def forward(self, Z):
        raise NotImplementedError("Overrride me!")

    @abc.abstractmethod
    def backward(self, dLdZ: np.ndarray):
        raise NotImplementedError("Overrride me!")
        

class LinearLayer(Layer):
    def __init__(self, layernum: int, nodes_in: int, nodes_out: int, activation: Activation, biases: bool):
        self.layernum = layernum # first weight matrix is 1, used to avoid matrix op
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.activation = activation
        self.biases = biases
        # self.lr = lr
        def weight_init(): return np.random.random()-0.5
        self.weight_init_method = weight_init

        self.W = np.array([[self.weight_init_method() for _ in range(nodes_out)] for _ in range(nodes_in)])
        if biases: 
            self.bias = np.array([[self.weight_init_method()] for _ in range(nodes_out)])
        

    def forward(self, X):
        self.forward_X = X
        Y = self.W.T@X
        self.forward_Y = Y
        if self.biases:
            Y += self.bias
        Z = self.activation.forward(Y)
        return Z
        

    def backward(self, dLdZ, lr):
        dLdY = self.activation.backward(self.forward_Y, dLdZ)
        dLdW = self.forward_X*(dLdY.T)
        if self.layernum > 0: # avoiding unnecesary calculation
            dLdZ = self.W@dLdY

        # update:
        self.W -= lr*dLdW
        self.bias -= lr*dLdY

        if self.layernum > 0: return dLdZ
