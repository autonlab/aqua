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
import torch
import sklearn
import copy
from sklearn.model_selection import train_test_split

# Base model imports
from aqua.models.presets import AqNet
from aqua.data import Aqdata, TestAqdata

# Cleaning model imports
from aqua.models.cleaning_modules import *
from aqua.configs import main_config, data_configs, model_configs


METHODS = main_config['methods']

class AqModel:
    def __init__(self, model,
                       architecture,
                       method, 
                       dataset,
                       device='cpu',
                       optimizer=None):
        weighted_loss = data_configs[dataset]['weighted_loss'] if ('weighted_loss' in data_configs[dataset] and data_configs[dataset]['weighted_loss']) else False

        self.model = AqNet(model, 
                           output_dim=data_configs[dataset]['out_classes'],
                           epochs=model_configs['base'][architecture]['epochs'],
                           batch_size=model_configs['base'][architecture]['batch_size'],
                           lr=model_configs['base'][architecture]['batch_size'],
                           lr_drops=model_configs['base'][architecture]['lr_drops'],
                           weighted_loss=weighted_loss,
                           device=device,
                           optimizer=optimizer)
        self.method = method

        # Add a wrapper over base model
        if method == 'cleanlab':
            self.wrapper_model = CleanLab(self.model)
        elif method == 'aum':
            self.wrapper_model = AUM(self.model, optimizer)
        elif method == "cincer":
            self.wrapper_model = CINCER(self.model, optimizer)
        elif method == 'simifeat':
            self.wrapper_model = SimiFeat(self.model)
        elif method == 'noisy':
            self.wrapper_model = self.model

    def find_label_issues(self, data_aq):
        if self.method == 'noisy':
            raise RuntimeError("get_cleaned_labels cannot be implemented with noisy method")
        
        cleaning_model_config = model_configs['cleaning'][self.method]
        label_issues = self.wrapper_model.find_label_issues(data_aq, **cleaning_model_config)
            
        # Label issues must be False if no issue, True if there is an issue
        return label_issues.astype(int)
    

    def _split_data(self, data_aq,
                          test_size = 0.25, 
                          random_state = 0):
        train_data_aq = copy.deepcopy(data_aq)
        val_data_aq = copy.deepcopy(data_aq)
        train_inds, val_inds = train_test_split(np.arange(data_aq.data.shape[0]),
                                                test_size=test_size,
                                                random_state=random_state)
        
        train_data_aq.set_inds(train_inds)
        val_data_aq.set_inds(val_inds)

        return train_data_aq, val_data_aq



class TrainAqModel(AqModel):
    def __init__(self, model, 
                    architecture, 
                    method, 
                    dataset, 
                    device='cpu', 
                    optimizer=None):
        super().__init__(model, architecture, method, dataset, device, optimizer)
        # Train should only support fit/fit_predict ?

    def fit(self, data_aq):
        self.model.fit(data_aq)

    def predict(self, data_aq):
        return self.model.predict(data_aq)

    def fit_predict(self, data_aq, return_val_labels=False):
        if return_val_labels:
            train_data_aq, val_data_aq = self._split_data(data_aq)
        else:
            train_data_aq = copy.deepcopy(data_aq)

        self.fit(train_data_aq)

        if not return_val_labels:
            return self.predict(train_data_aq)
        else:
            return self.predict(val_data_aq), val_data_aq.labels


# TODO (vedant, mononito) : please review this
class TestAqModel(AqModel):
    def __init__(self, method, modality, model):
        super().__init__(modality, method)
        self.model = model

    def predict(self, data_aq):
        return self.model.predict(data_aq)



        
