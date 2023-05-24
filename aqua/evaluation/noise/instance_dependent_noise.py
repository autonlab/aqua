import numpy as np
from .noise_abc import SyntheticNoise
from copy import deepcopy
from scipy.stats import truncnorm
from scipy.special import softmax

class InstanceDependentNoise(SyntheticNoise):
    def __init__(self, n_classes:int=2, noise_rate:float=0.2, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.p = noise_rate
                
    def add_noise(self, X:np.array, y:np.array):
        if y.squeeze().ndim != 1:
            y = np.argmax(y, axis=1) # In case the inputs are one hot already

        featurized_X = self.get_featurized_dataset(X) # TODO
        self.N, self.S = featurized_X.shape # Number of data points and features

        # Sample instance flip rates qn from the truncated normal distribution
        q_n = truncnorm.rvs(a=0, b=1, loc=self.p, scale=0.1, size=self.N)

        # Sample W from the standard normal distribution
        W = np.random.normal(loc=0, scale=1, size=(self.S, self.n_classes))

        # Generate instance dependent flip rates. The size of p is N x K.
        P = np.matmul(featurized_X, W)            
        for n in range(self.N): P[n, y[n]] = -1 * np.inf # Only consider entries that are different from the true label
        P = q_n * softmax(P, axis=1)

        noisy_y = []
        for n in enumerate(range(self.N)):
            P[n, y[n]] = 1 - q_n[n] # Keep clean w.p. 1 − q_n
            
            # Let q_n be the probability of getting a wrong label
            # Randomly choose a label from the label space as noisy label \tilde y_n according to p
            noisy_y.append(np.random.choice(a=np.arange(self.n_classes), size=1, p=P[n, :])[0])
            
        noisy_y = np.asarray(noisy_y)
        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y
    
    def get_featurized_dataset(self, X:np.ndarray):
        # Featurize the dataset using trained model
        # return featurized_X
        raise NotImplementedError()

