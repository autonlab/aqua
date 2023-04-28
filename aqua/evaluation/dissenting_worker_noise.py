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



