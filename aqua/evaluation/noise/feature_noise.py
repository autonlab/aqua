import numpy as np
from copy import deepcopy
import random
from .noise_abc import SyntheticNoise
from timeseries_noise import InjectAnomalies

class FeatureNoise(SyntheticNoise):
    def __init__(self, modality:str="image", **kwargs):
        super().__init__()

        self.modality = modality

        if self.modality == "image":
            self.mean = kwargs["mean"] if "mean" in kwargs else 0
            self.sigma = kwargs["sigma"] if "sigma" in kwargs else 0.01
        elif self.modality == "timeseries":
            self.ts_noiseobj = InjectAnomalies() 
            self.anomaly_type = kwargs["anomaly_type"] if "anomaly_type" in kwargs else "noise"

    def add_image_noise(self, X:np.array, y:np.array):
        N, row, col, ch = X.shape
        noise = np.random.normal(self.mean, self.sigma, (N, row, col, ch))
        noisy_X = X + noise
        return noisy_X, y

    def add_timeseries_noise(self, X:np.array, y:np.array, anomaly_type:str):
        N, ch, t = X.shape
        noisy_X_list, noisy_y_list = [], []
        for i in range(N):
            try:
                noisy_X, anomaly_size, anomaly_labels = self.ts_noiseobj.inject_anomalies(X[i], anomaly_type=anomaly_type)
                noisy_X_list.append(noisy_X)
            except:
                noisy_X_list.append(X[i])
        noisy_X = np.array(noisy_X_list)

        return noisy_X, y

    def add_noise(self, X:np.array, y:np.array):
        if self.modality == "image":
            noisy_X, y = self.add_image_noise(X, y)
        elif self.modality == "timeseries":
            noisy_X, y = self.add_timeseries_noise(X, y, self.anomaly_type)
        return noisy_X, y