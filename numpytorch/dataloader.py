import numpy as np

class MNISTDataLoader:
    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, testing_data: np.ndarray, testing_labels: np.ndarray):
        """ 
        training_data and testing_data must be of form TODO.
        training_labels and testing_labels must be of form TODO.
        
        """
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels
    
    def training_set(self):
        return self.training_data, self.training_labels
    
    def testing_set(self):
        return self.testing_data, self.testing_labels
