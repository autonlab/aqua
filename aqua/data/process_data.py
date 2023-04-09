import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from aqua.utils import load_single_datapoint

class Aqdata(Dataset):
    def __init__(self, data, ground_labels,
                 **kwargs):
        self.data = data
        self.labels = ground_labels
        self.corrected_labels = kwargs['corrected_labels'] if 'corrected_labels' in kwargs else None
        self.lazy_load = kwargs['lazy_load'] if 'lazy_load' in kwargs else False
        self.kwargs = kwargs
        self.attention_masks = None

        # Additional capability to add noise
        self.noise_rate = kwargs['noise_rate'] if 'noise_rate' in kwargs else 0.0
        self.noise_type = kwargs['noise_type'] if 'noise_type' in kwargs else None
        self.noise_prior = None
        self.noise_or_not = np.array([False]*data.shape[0]) # Keeps track of labels purposefully corrupted by noise
        # TODO : (mononito/arvind) : please make sure you're updating `noise_or_not` once labels are corrupted with noise
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
        if self.lazy_load:
            ret_data = load_single_datapoint(self.data[idx])
        else:
            ret_data = self.data[idx]
        return_args = [ret_data, self.labels[idx], idx]
        if self.corrected_labels is not None:
            return_args.append(self.corrected_labels[idx])
        else:
            return_args.append('None')
        
        # Add all misc model specific kwargs here
        misc_kwargs = {}
        if self.attention_masks is not None: misc_kwargs['attention_mask'] = self.attention_masks[idx]
        return_args.append(misc_kwargs)       
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

        # Add all misc model specific kwargs here
        misc_kwargs = {}
        if self.attention_masks is not None: misc_kwargs['attention_mask'] = self.attention_masks[idx]
        return_args.append(misc_kwargs)  
        
        return tuple(return_args)