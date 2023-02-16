import os, torch
import numpy as np
import pandas as pd
from aum import AUMCalculator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader

"""
Simple wrapper around a base ML model for the 
AUM cleaning model.
"""
class AUM:
    def __init__(self, model):
        self.model = model
        self.dthr = None
        self.orig_dim = self.model.output_dim

        self.model.reinit_model(self.model.model_type, self.model.output_dim+1)


    def _fit_get_aum(self, thr_inds, alpha=0.99):
        self._aum_calculator = AUMCalculator(os.getcwd())
        train_metrics = self.model.get_training_metrics()
        for i in range(len(train_metrics['output'])):
            self._aum_calculator.update(train_metrics['output'][i],
                                        train_metrics['target'][i],
                                        train_metrics['sample_id'][i])
        self._aum_calculator.finalize()
        aum_file = pd.read_csv(os.path.join(os.getcwd(), 'aum_values.csv'))
        thresh = np.percentile(aum_file.iloc[thr_inds]['aum'].values, 99)
        #d_thresh = aum_file.iloc[thr_inds]['aum'].values
        mask = np.array([True]*aum_file.shape[0])  # Selects train indices only, discards THR indices
        mask[thr_inds] = False

        return np.array(aum_file.index)[mask][aum_file['aum'].values[mask] < thresh]
        

    def find_label_issues(self, data, labels):
        # Refer to paper for training strategy

        # Randomly assign N/(c+1) data as the (c+1)th class
        label_val = np.unique(labels).max()+1
        N, c = data.shape[0], np.unique(labels).shape[0]+1
        
        rand_inds = np.random.randint(0, N, size=2*(N//c))

        # Pass 1
        labels_step_1 = labels.copy()
        labels_step_1[rand_inds[:(N//c)]] = label_val
        self.fit(data, labels_step_1)
        incorrect_labels_idx = self._fit_get_aum(rand_inds[:(N//c)])
        #print(incorrect_labels_idx.shape)

        # Pass 2
        self.model.reinit_model(self.model.model_type, self.model.output_dim)
        labels_step_2 = labels.copy()
        labels_step_2[rand_inds[(N//c):]] = label_val
        self.fit(data, labels_step_2)
        incorrect_labels_idx_thresh = self._fit_get_aum(rand_inds[(N//c):])
        total_incorrect_labels = np.union1d(incorrect_labels_idx, incorrect_labels_idx_thresh)
        #print(incorrect_labels_idx_thresh.shape)
        #print(total_incorrect_labels.shape)

        mask = np.array([False]*data.shape[0])  # Selects train indices only, discards THR indices
        mask[total_incorrect_labels] = True

        # Re-instantiate the model with correct number of output neurons
        self.model.reinit_model(self.model.model_type, self.orig_dim)
        return mask
    
    def fit(self, data, labels):        
        return self.model.fit(data, labels, 
                              lr_tune=True,
                              early_stop=True)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)


class CINCER:
    def __init__(self):
        raise KeyboardInterrupt 