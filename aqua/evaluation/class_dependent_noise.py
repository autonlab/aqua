import numpy as np
from .noise_abc import SyntheticNoise
from typing import Union, List
import numpy.typing as npt 

class ClassDependentNoise(SyntheticNoise):
    def __init__(self, 
                 n_classes:int=2, 
                 noise_rate:Union[float, List, npt.NDArray]=[0.1, 0.4]):
        super().__init__()
        """
        Notes: 
        There are two ways to inject class dependent noise: 
        (1) From the confusion matrix of learned models
        (2) arbitrary structured noise transition matrix 
        """

        self.n_classes = n_classes
        self.noise_transition_matrix = np.identity(n=self.n_classes)
        self.p = noise_rate
        
        # Make noise transition matrix
        self.make_noise_transition_matrix()

    def make_noise_transition_matrix(self):
        self.noise_transition_matrix =\
              (1 - self.p)*self.noise_transition_matrix + (self.p/(self.n_classes - 1))*(1-self.noise_transition_matrix)
        
    def add_noise(self, X:np.array, y:np.array):
        _y = self.make_labels_one_hot(self.check_process_inputs(y))
        N = y.shape[0]
        sampling_probabilities = np.matmul(_y, self.noise_transition_matrix.T)
        noisy_y = np.zeros((_y.shape[0], 1))
        
        for i in range(N): # Can probably be made efficient
            noisy_y[i] = np.argmax(np.random.multinomial(n=1, pvals=sampling_probabilities[i, :], size=1))
        
        noisy_y = noisy_y.squeeze().astype(int)

        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y

