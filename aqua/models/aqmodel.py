import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split

# Aqua imports
import cleanlab as cl

from aqua.models.presets import ImageNet
from aqua.data import Aqdata, TestAqdata

output_dict = {
    "cifar10" : 10
}

class AqModel:
    def __init__(self, modality, 
                       method, 
                       dataset,
                       device='cpu'):
        if modality == 'image':
            self.model = ImageNet('resnet34',
                                 epochs=6,
                                 output_dim=output_dict[dataset],
                                 device=device)
        else:
            raise RuntimeError(f"Incorrect modality: {modality}")
        self.method = method
        self.wrapper_model = self.model

        # Add a wrapper over base model
        if method == 'cleanlab':
            self.wrapper_model = cl.classification.CleanLearning(self.model)

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
        if self.method == 'cleanlab':
            return self.wrapper_model.predict(data)

class TrainAqModel(AqModel):
    def __init__(self, modality, method, dataset, device='cpu'):
        super().__init__(modality, method, dataset, device)
        # Train should only support fit/fit_predict ?

    def fit(self, data, labels):
        if self.method == 'cleanlab':
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


class TestAqModel(AqModel):
    def __init__(self, method, modality, model):
        super().__init__(modality, method)
        self.model = model

    def predict(self, data):
        return super().predict(data)



        
