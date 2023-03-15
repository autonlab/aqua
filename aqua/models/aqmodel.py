import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split

# Base model imports
from aqua.models.presets import ImageNet, TextNet
#from aqua.models.cleaning_models import AUM, CINCER, ActiveLabelCleaning, SimiFeat
from aqua.data import Aqdata, TestAqdata

# Cleaning model imports
from aqua.models.modules import *
from aqua.configs import main_config, data_configs, model_configs


METHODS = main_config['methods']

class AqModel:
    def __init__(self, modality, 
                       architecture,
                       method, 
                       dataset,
                       device='cpu'):
        if modality == 'image':
            self.model = ImageNet(architecture,
                                 epochs=1,
                                 output_dim=data_configs[dataset]['out_classes'],
                                 device=device)
        elif modality == 'text':
            self.model = TextNet(architecture,
                                 epochs=1,
                                 output_dim=data_configs[dataset]['out_classes'],
                                 device=device)
        else:
            raise RuntimeError(f"Incorrect modality: {modality}")
        self.method = method

        # Add a wrapper over base model
        if method == 'cleanlab':
            self.wrapper_model = CleanLab(self.model)
        elif method == 'aum':
            self.wrapper_model = AUM(self.model)
        elif method == "cincer":
            self.wrapper_model = CINCER(self.model)
        elif method == "active_label_cleaning":
            self.wrapper_model = ActiveLabelCleaning(self.model)
        elif method == 'simifeat':
            self.wrapper_model = SimiFeat(self.model)
        elif method == 'noisy':
            self.wrapper_model = self.model

    def get_cleaned_labels(self, data, label):
        if self.method == 'noisy':
            raise RuntimeError("get_cleaned_labels cannot be implemented with noisy method")
        
        cleaning_model_config = model_configs['cleaning'][self.method]
        label_issues = self.wrapper_model.find_label_issues(data, label, **cleaning_model_config)
            
        # Label issues must be False if no issue, True if there is an issue
        data, label = data[~label_issues], label[~label_issues]
        return data, label, label_issues
    

    def _split_data(self, data, 
                          labels,
                          test_size = 0.25, 
                          random_state = 0):
        data_train, data_val, labels_train, labels_val = train_test_split(data,
                                                                          labels,
                                                                          test_size=test_size,
                                                                          random_state=random_state)

        return data_train, data_val, labels_train, labels_val


    def predict(self, data, model=None):
        if model is not None:
            self.wrapper_model = model
        if self.method in ['cleanlab', 'noisy']:
            # TODO: will all label error methods follow sklearn classifier schema?
            return self.wrapper_model.predict(data)



class TrainAqModel(AqModel):
    def __init__(self, modality, architecture, method, dataset, device='cpu'):
        super().__init__(modality, architecture, method, dataset, device)
        # Train should only support fit/fit_predict ?

    def fit(self, data, labels):
        if self.method in ['cleanlab', 'noisy']:
            self.wrapper_model.fit(data, labels)

    def predict(self, data):
        return super().predict(data)

    def fit_predict(self, data, labels, return_val_labels=False):
        train_data, val_data, train_labels, val_labels = self._split_data(data, 
                                                                          labels)

        self.fit(train_data, train_labels)

        if not return_val_labels:
            return self.predict(val_data)
        else:
            return self.predict(val_data), val_labels


# TODO (vedant, mononito) : please review this
class TestAqModel(AqModel):
    def __init__(self, method, modality, model):
        super().__init__(modality, method)
        self.model = model

    def predict(self, data):
        return super().predict(data)



        
