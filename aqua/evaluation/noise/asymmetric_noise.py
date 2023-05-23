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

