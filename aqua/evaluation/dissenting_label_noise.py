import numpy as np
from copy import deepcopy
import random
from .noise_abc import SyntheticNoise

class DissentingLabelNoise(SyntheticNoise):
    def __init__(self, n_classes:int=2, noise_rate:float=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.p = noise_rate

    def add_noise(self, X:np.array, y:np.array, annotator_y:np.array):
        n_classes = 10
        N = X.shape[0]
        
        annotator_label_set = np.apply_along_axis(set, axis=0, arr=annotator_y)

        #y = labels['aggre_label']
        noisy_y = deepcopy(y)

        sample_indices = np.arange(N)
        np.random.shuffle(sample_indices)
        label_errors = 0
        for i in range(N):
            idx = sample_indices[i]
            if len(annotator_label_set[idx]) > 1:
                noisy_y[idx] = random.choice(list(annotator_label_set[idx] - {y[idx]}))
                label_errors += 1
            if (label_errors/N) >= self.p:
                break
                
        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y



