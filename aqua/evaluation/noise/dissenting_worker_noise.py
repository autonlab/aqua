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
from copy import deepcopy
import random
from .noise_abc import SyntheticNoise

class DissentingWorkerNoise(SyntheticNoise):
    def __init__(self, n_classes:int=2, noise_rate:float=0.2, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.p = noise_rate
        self.multi_annotator = True

    def add_noise(self, X:np.array, y:np.array, annotator_y:np.array):
        n_classes = self.n_classes
        N = X.shape[0]

        noisy_y = deepcopy(y)

        sample_indices = np.arange(N)
        label_errors = 0
        workers = list(range(annotator_y.shape[0]))
        random.shuffle(workers)

        for worker in workers:
            np.random.shuffle(sample_indices)
            for idx in sample_indices:
                if annotator_y[worker][idx] != y[idx] and noisy_y[idx] == y[idx]:
                    noisy_y[idx] = annotator_y[worker][idx]
                    label_errors += 1
                if (label_errors/N) >= self.p:
                    break
            if (label_errors/N) >= self.p:
                break

        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y



