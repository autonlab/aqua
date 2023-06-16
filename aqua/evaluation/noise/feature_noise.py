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