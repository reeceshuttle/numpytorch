import numpy as np

import numpytorch as npt

class NeuralNetwork:
    def __init__(self, network_specification: dict):
        self.layers: list[npt.Layer] = []
        for layer_num, layer_spec in enumerate(network_specification):
            self.layers.append(layer_spec.layer_type(layernum=layer_num, 
                                                     nodes_in=layer_spec.nodes_in,
                                                     nodes_out=layer_spec.nodes_out,
                                                     activation=layer_spec.activation,
                                                     biases=layer_spec.biases))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLdZ: np.ndarray, lr: float) -> np.ndarray:
        for layer in reversed(self.layers):
            dLdZ = layer.backward(dLdZ, lr)