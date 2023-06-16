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
from .noise_abc import SyntheticNoise

class UniformNoise(SyntheticNoise):
    def __init__(self, n_classes:int=2, noise_rate:float=0.2, **kwargs):
        super().__init__()
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

