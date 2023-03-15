import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Aqdata(Dataset):
    def __init__(self, data, ground_labels, 
                 corrected_labels=None,
                 noise_rate=0.0,
                 noise_type=None,
                 **kwargs):
        self.data = data
        self.labels = ground_labels
        self.corrected_labels = corrected_labels
        self.kwargs = kwargs
        self.attention_masks = None

        # Additional capability to add noise
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.noise_prior = None
        self.noise_or_not = np.array([False]*data.shape[0]) # Keeps track of labels purposefully corrupted by noise
        if 'attention_mask' in kwargs: self.attention_masks = kwargs['attention_mask']

    def clean_data(self, label_issues):
        self.data = self.data[~label_issues]
        self.labels = self.labels[~label_issues]
        if self.attention_masks is not None: 
            self.attention_masks = self.attention_masks[~label_issues]

    def set_inds(self, inds):
        self.data = self.data[inds]
        self.labels = self.labels[inds]
        if self.attention_masks is not None:
            self.attention_masks = self.attention_masks[inds]
    
    def add_noise(self, data, labels):
        raise NotImplementedError

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Pytorch models will have to be handled separately here
        return_args = [self.data[idx], self.labels[idx], idx]
        if self.corrected_labels is not None:
            return_args.append(self.corrected_labels[idx])
        else:
            return_args.append('None')

        if self.attention_masks is not None: return_args.append(self.attention_masks[idx])        
        return tuple(return_args)
        


class TestAqdata(Dataset):
    def __init__(self, data, **kwargs):
        self.data = data
        self.attention_masks = None
        if 'attention_mask' in kwargs: self.attention_masks = kwargs['attention_mask']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return_args = [self.data[idx], idx]
        if self.attention_masks is not None: return_args.append(self.attention_masks[idx])
        return tuple(return_args)