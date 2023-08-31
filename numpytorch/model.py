import numpy as np

import numpytorch as npt

class NeuralNetwork:
    def __init__(self, nn_sequential: list[npt.Layer]):
        self.layers = nn_sequential
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLd_: np.ndarray, lr: float) -> np.ndarray:
        for layer in reversed(self.layers):
            dLd_ = layer.backward(dLd_, lr)
            if dLd_ == "exit backprop":
                break