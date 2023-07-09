import numpy as np
import abc

class Activation:
    @abc.abstractmethod
    def forward(self, Y: np.ndarray) -> np.ndarray:
        "Forward pass through activation during inference."
        raise NotImplementedError("Override me!")
    
    @abc.abstractmethod
    def backward(self, Y: np.ndarray, dLdZ: np.ndarray) -> np.ndarray:
        """
        backward pass through activation during backprop.
        Returns dLdY.
        """
        raise NotImplementedError("Overrride me!")

class Identity(Activation):
    def forward(Y):
        return Y

    def backward(Y, dLdZ):
        return dLdZ

class ReLU(Activation):
    def forward(Y):
         return Y * (Y>0)

    def backward(Y, dLdZ):
        diagonal_vals = 1*(Y>0)
        dZdY = np.diag(np.squeeze(diagonal_vals))
        dLdY = dZdY@dLdZ
        return dLdY

class Sigmoid(Activation):
    def forward(Y):
        return 1/(1+np.exp(-Y))

    def backward(Y, dLdZ):
        diagonal_vals = Sigmoid.forward(Y)*(1-Sigmoid.forward(Y)) # store this for cheaper computation? maybe even forward pass?
        dZdY = np.diag(np.squeeze(diagonal_vals))
        dLdY = dZdY@dLdZ
        return dLdY

class Softmax(Activation):
    def forward(Y): # store forward pass for cheaper computation?
        num = np.exp(Y)
        denom = np.sum(num)
        return num/denom

    def backward(Y, dLdZ):
        dZdY = np.diag(np.squeeze(Softmax.forward(Y)))-(Softmax.forward(Y)@(Softmax.forward(Y).T))
        dLdY = dZdY@dLdZ
        return dLdY