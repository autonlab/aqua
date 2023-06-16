# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder

class SyntheticNoise(ABC):    
    def __init__(self):
        self._noise_or_not = None
        self.multi_annotator = False

    @property
    def noise_or_not(self):
        return self._noise_or_not
    
    @noise_or_not.setter 
    def noise_or_not(self, value:np.ndarray):
        self._x = value
    
    @noise_or_not.deleter 
    def noise_or_not(self):
        del self._x

    @abstractmethod
    def add_noise(self, X:np.ndarray, y:np.ndarray):
        """Add noise to input data (X, y)
        """
        pass

    def make_noise_transition_matrix(self):
        """Make noise transition matrix to inject noise

        A noise transition matrix M is a K x K matrix for k-class
        classification problems. M[i, j] denotes the probability
        that the observed (noisy) label is i, when the ground 
        truth (true) label is j.
        """
        pass
    
    def make_labels_one_hot(self, y:np.ndarray):
        encoder = OneHotEncoder(sparse=False, sparse_output=False) # Maybe change
        return encoder.fit_transform(y)
    
    def check_process_inputs(self, y:np.ndarray):
        if len(y.shape) == 1: 
            y = y.reshape((-1, 1))
        else:
            raise AttributeError('Labels must be of shape (-1,) or (-1,1)') # This wouldn't work for multi-annotator settings
        return y

    def estimate_noise_rate(self, y:np.ndarray, noisy_y:np.ndarray):
        """Estimates the rate of added noise
        """
        # Check inputs
        y = self.check_process_inputs(y)
        noisy_y = self.check_process_inputs(noisy_y)
        assert y.shape == noisy_y.shape, "y and noisy_y shape mismatch"

        return np.mean(y != noisy_y)

    def estimate_noise_transition_matrix(self, y:np.ndarray, noisy_y:np.ndarray):
        """Estimates the noise transition matrix based on
        added noise and existing labels
        """
        # Check inputs
        y = self.check_process_inputs(y)
        noisy_y = self.check_process_inputs(noisy_y)
        assert y.shape == noisy_y.shape, "y and noisy_y shape mismatch"
        
        n_classes = self.infer_n_classes(y)
        n = y.shape[0]

        estimated_noise_transition_matrix = np.zeros((n_classes, n_classes))

        for iter in range(n):
            estimated_noise_transition_matrix[noisy_y[iter], y[iter]]+=1
        
        # Normalize by rows
        estimated_noise_transition_matrix =\
              estimated_noise_transition_matrix/np.sum(estimated_noise_transition_matrix, axis=1, keepdims=True)

        return estimated_noise_transition_matrix
    
    def infer_n_classes(self, y:np.ndarray):
        """Infer number of label classes
        """
        return len(np.unique(y))