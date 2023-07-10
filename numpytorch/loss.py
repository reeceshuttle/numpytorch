import numpy as np
import abc

class Loss:
    @abc.abstractmethod
    def loss(self, estimate: np.ndarray, actual:np.ndarray) -> float:
        raise NotImplementedError("Override me!")
    
    @abc.abstractmethod
    def gradient(self, estimate, actual) -> np.ndarray:
        raise NotImplementedError("Override me!")


class SquaredLoss(Loss):
    def loss(estimate, actual):
        return np.sum((estimate-actual)**2)
    
    def gradient(estimate, actual):
        dLdZ = 2*(estimate-actual)
        return dLdZ
    
class NLLM(Loss):
    def loss(estimate, actual):
        return -np.sum(actual*np.log(estimate))
    
    def gradient(estimate, actual):
        return -actual/estimate