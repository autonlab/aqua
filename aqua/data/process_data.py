import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset

from aqua.utils import load_single_datapoint
from aqua.evaluation.noise import *
from aqua.configs import main_config

class Aqdata(Dataset):
    def __init__(self, data, ground_labels,
                 **kwargs):
        self.data = data
        self.labels = ground_labels
        self.n_classes = pd.unique(self.labels).shape[0]
        self.corrected_labels = kwargs['corrected_labels'] if 'corrected_labels' in kwargs else None
        self.lazy_load = kwargs['lazy_load'] if 'lazy_load' in kwargs else False
        self.kwargs = kwargs
        self.attention_masks = None

        # Additional capability to add noise
        self._noise_rate = kwargs['noise_rate'] if 'noise_rate' in kwargs else 0.0
        self.noise_type = kwargs['noise_type'] if 'noise_type' in kwargs else 'uniform'
        self.noise_prior = None
        self.noise_or_not = np.array([False]*data.shape[0]) # Keeps track of labels purposefully corrupted by noise
        # TODO : (mononito/arvind) : please make sure you're updating `noise_or_not` once labels are corrupted with noise
        if 'attention_mask' in kwargs: self.attention_masks = kwargs['attention_mask']

        # Multi-Annotator Datasets
        self.annotator_labels = kwargs['annotator_labels'] if 'annotator_labels' in kwargs else None

        # Add noise if noise_rate is > 0.0
        if self.noise_rate > 0.0:
            self.data, self.labels = self.add_noise(self.data, self.labels)
    
    @property
    def noise_rate(self):
        return self._noise_rate
    
    @noise_rate.setter
    def noise_rate(self, noise_args):
        noise_kwargs = {}
        if isinstance(noise_args, (tuple, list)):
            noise_rate, noise_kwargs = noise_args[0], noise_args[1]
        else:
            noise_rate = noise_args
        self._noise_rate = noise_rate
        self.data, self.labels = self.add_noise(self.data, self.labels, noise_kwargs)

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
    
    def add_noise(self, data, labels, noise_kwargs={}):
        logging.info(f"Adding noise with noise rate: {self.noise_rate}")
    
        if self.noise_type == 'uniform':
            self.noise_model = UniformNoise(self.n_classes, self.noise_rate)
        elif self.noise_type == 'dissenting_worker':
            self.noise_model = DissentingWorkerNoise(self.n_classes, self.noise_rate)
        elif self.noise_type == 'dissenting_label':
            self.noise_model = DissentingLabelNoise(self.n_classes, self.noise_rate)
        elif self.noise_type == 'asymmetric':
            self.noise_model = AsymmetricNoise(self.n_classes, self.noise_rate)
        elif self.noise_type == 'class_dependent':
            self.noise_model = ClassDependentNoise(self.n_classes, 
                                                  model=noise_kwargs['model'],
                                                  data=noise_kwargs['data'],
                                                  device=main_config['device'],
                                                  batch_size=noise_kwargs['batch_size'])
        elif self.noise_type == 'instance_dependent':
            self.noise_model = InstanceDependentNoise(self.n_classes,
                                                     noise_rate=self.noise_rate,
                                                     model=noise_kwargs['model'],
                                                     device=main_config['device'],
                                                     batch_size=noise_kwargs['batch_size'])
        else:
            raise RuntimeError(f"Incorrect noise type provided: {self.noise_type}, currently supported noise types: uniform, dissenting_worker, dissenting_label")
            
        if self.noise_model.multi_annotator:
            noisy_X, noisy_y = self.noise_model.add_noise(X=data, y=labels, annotator_y=self.annotator_labels)
        else:
            noisy_X, noisy_y = self.noise_model.add_noise(X=data, y=labels)
        self.noise_or_not = self.noise_model.noise_or_not

        return noisy_X, noisy_y

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