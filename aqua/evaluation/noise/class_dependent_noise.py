import numpy as np
from .noise_abc import SyntheticNoise
from typing import Union, List, Optional
import numpy.typing as npt 
#from aqua.data.process_data import Aqdata
#from aqua.models.aqmodel import AqModel
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class ClassDependentNoise(SyntheticNoise):
    def __init__(self, 
                 n_classes:int=2, 
                 noise_type:str='confusion_matrix', 
                 noise_transition_matrix:Optional[Union[float, List, npt.NDArray]]=None,
                 model = None,
                 data = None,
                 device:Optional[str]='cpu',
                 batch_size:Optional[int]=32):
        super().__init__()
        """
        Notes: 
        There are two ways to inject class dependent noise: 
        (1) From the confusion matrix of learned models (type == 'confusion_matrix')
        (2) arbitrary structured noise transition matrix (type == 'noise_rate')
        """
        
        self._noise_type_options = ['confusion_matrix', 'noise_rate']
        self.n_classes = n_classes
        self.noise_transition_matrix = np.identity(n=self.n_classes)
        self.p = noise_transition_matrix
        self.noise_type = noise_type
        
        # Arguments when noise_type == 'confusion_matrix'
        self.model = model
        self.data = data
        self.device = device
        self.batch_size = batch_size
        
        # Check input arguments
        self._check_arguments()

        # Make noise transition matrix
        if self.noise_type == 'confusion_matrix':
            self.noise_transition_matrix = self.estimate_confusion_matrix()
        else:
            self.noise_transition_matrix = noise_transition_matrix

    def _check_arguments(self):
        if self.noise_type not in self._noise_type_options:
            raise ValueError(f'noise_type must be one of {self._noise_type_options} but {self.noise_type} was passed!')
        
    def make_noise_transition_matrix(self):
        pass
        
    def add_noise(self, X:np.array, y:np.array):
        _y = self.make_labels_one_hot(self.check_process_inputs(y))
        N = y.shape[0]
        sampling_probabilities = np.matmul(_y, self.noise_transition_matrix.T)
        noisy_y = np.zeros((_y.shape[0], 1))
        
        for i in range(N): # Can probably be made efficient
            probs = sampling_probabilities[i,:].astype('float64')
            probs /= np.sum(probs)
            noisy_y[i] = np.argmax(np.random.multinomial(n=1, pvals=probs, size=1))
        
        noisy_y = noisy_y.squeeze().astype(int)

        self._noise_or_not = (y != noisy_y).astype(int)

        return X, noisy_y
    
    def estimate_confusion_matrix(self):
        logging.info('Estimating confusion matrix')        
        
        model = self.model.model
        model.eval()

        dataloader = DataLoader(self.data, 
                                batch_size=self.batch_size, 
                                shuffle=False, 
                                num_workers=1)

        y_pred = []
        y_true = []
    
        # Send model to device
        model = model.to(self.device)
        with torch.no_grad():
            for (data, target, _, _, _) in tqdm(dataloader, desc='Class Dependent Noise:'):

                data = data.to(self.device)
                target = target.to(self.device)
                preds = model(data)

                y_pred.append(preds.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy().reshape((-1, 1)))

        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = np.argmax(y_pred, axis=1, keepdims=True)
        y_true = np.concatenate(y_true, axis=0)

        cmatrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
        cmatrix = np.around(cmatrix, decimals=3)

        logging.info(f'Confusion matrix:\n{cmatrix}')
        
        return cmatrix

