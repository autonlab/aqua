import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder

class SyntheticNoise(ABC):    
    def __init__():
        pass
    
    @abstractmethod  # Decorator to define an abstract method
    def make_noise_transition_matrix(self):
        """Make noise transition matrix to 
        """
        pass

    @abstractmethod
    def add_noise(self, X:np.array, y:np.array):
        """Add noise to input data (X, y)
        """
        pass
    
    @abstractmethod
    def make_labels_one_hot(self, y):
        encoder = OneHotEncoder(sparse=False, sparse_output=False) # Maybe change
        return encoder.fit_transform(y)
    
    @abstractmethod
    def check_process_inputs(self, X, y):
        if len(y.shape) == 1: 
            y = y.reshape((-1, 1))
        return y
        



