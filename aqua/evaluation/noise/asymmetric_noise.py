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
from copy import deepcopy

class AsymmetricNoise(SyntheticNoise):
    def __init__(self, n_classes:int=2, noise_rate:float=0.2, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.p = noise_rate
                
    def add_noise(self, X:np.array, y:np.array):
        if y.squeeze().ndim != 1:
            y = np.argmax(y, axis=1) # In case the inputs are one hot already

        noisy_y = deepcopy(y)
        noisy_idxs = np.random.choice(a=np.arange(len(y)), replace=False, size=int(self.p*len(y)))
        
        noisy_y[noisy_idxs] = (noisy_y[noisy_idxs] + 1)%self.n_classes
        
        noisy_y = noisy_y.squeeze().astype(int)

        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y

