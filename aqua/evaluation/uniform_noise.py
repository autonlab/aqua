import numpy as np
from .noise_abc import SyntheticNoise

class UniformNoise(SyntheticNoise):
    super().__init__()
    def __init__(self, n_classes:int=2):
        self.n_classes = n_classes
        self.noise_transition_matrix = np.identity(n=self.n_classes)
        self.p = 0

    def make_noise_transition_matrix(self, p:float):
        self.p = p
        self.noise_transition_matrix =\
              p*self.noise_transition_matrix + ((1 - p)/(self.n_classes - 1))*(1-self.noise_transition_matrix)
        
    def add_noise(self, X:np.array, y:np.array):
        y = super().make_labels_one_hot(super().check_process_inputs(y))
        N = y.shape[0]
        sampling_probabilities = np.matmul(y, self.noise_transition_matrix)
        noisy_y = np.zeros((y.shape[0], 1))
        
        for i in range(N): # Can probably be made efficient
            noisy_y[i] = np.random.multinomial(n=1, pvals=sampling_probabilities[i, :], size=1)

        return noisy_y
        
        
    



