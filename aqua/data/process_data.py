import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Aqdata(Dataset):
    def __init__(self, data, ground_labels, 
                 corrected_labels=None,
                 noise_rate=0.0,
                 noise_type=None):
        self.data = data
        self.labels = ground_labels
        self.corrected_labels = corrected_labels

        # Additional capability to add noise
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.noise_prior = None
        self.noise_or_not = np.array([False]*data.shape[0]) # Keeps track of labels purposefully corrupted by noise
    
    def add_noise(self, data, labels):
        raise NotImplementedError

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Pytorch models will have to be handled separately here
        if self.corrected_labels is not None:
            return self.data[idx], self.labels[idx], idx, self.corrected_labels[idx]
        else:
            return self.data[idx], self.labels[idx], idx, 'None'


class TestAqdata(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], idx