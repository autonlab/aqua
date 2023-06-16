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

import torch, copy, logging
import numpy as np
import pandas as pd

from .sim_utils import noniterate_detection
from . import global_var as global_var

from aqua.data import Aqdata
from torch.utils.data import DataLoader

class SimiFeat:
    def __init__(self, model):
        self.model = model

    def _noniterate_detect(self, data_aq, desc='') -> np.ndarray:
        labels = data_aq.labels
        num_classes = np.unique(labels).shape[0]
        N = labels.shape[0]
        self.config.cnt = min(self.config.cnt, N//4)

        self.config.num_classes = num_classes
        trainloader = DataLoader(data_aq,
                                 batch_size=self.model.batch_size,
                                 shuffle=True,
                                 num_workers=4)
        
        sel_noisy_rec = []
        sel_clean_rec = np.zeros((self.config.num_epoch, N))
        sel_times_rec = np.zeros(N)
        global_var._init()
        global_var.set_value('T_init', None)
        global_var.set_value('p_init', None)
        for epoch in range(self.config.num_epoch):
            record = [[] for _ in range(num_classes)]

            for i_batch, (feature, label, index, _, data_kwargs) in enumerate(trainloader):
                feature, label = feature.to(self.model.device).float(), label.to(self.model.device)
                with torch.no_grad():    
                    _, extracted_feat = self.model.model(feature, return_feats=True)
                for i in range(extracted_feat.shape[0]):
                    record[label[i]].append({'feature':extracted_feat[i].detach().cpu(), 'index':index[i]})

            if self.config.method == 'both':
                #rank1 + mv
                self.config.method = 'rank1'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, data_aq, 
                                                                    sel_noisy=sel_noisy_rec.copy())

                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5
                sel_times_rec[np.array(sel_idx)] += 0.5

                self.config.method = 'mv'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, data_aq, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5

                self.config.method = 'both'
                sel_times_rec[np.array(sel_idx)] += 0.5
            else:
                # use one method
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, data_aq, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                if self.config.num_epoch > 1:
                    sel_clean_rec[epoch][np.array(sel_clean)] = 1
                    sel_times_rec[np.array(sel_idx)] += 1

            aa = np.sum(sel_clean_rec[:epoch + 1], 0) / sel_times_rec
            nan_flag = np.isnan(aa)
            aa[nan_flag] = 0

            #sel_clean_summary = np.round(aa).astype(bool)
            sel_noisy_summary = np.round(1.0 - aa).astype(bool)
            sel_noisy_summary[nan_flag] = False

            return sel_noisy_summary

    def find_label_issues(self, data_aq, **kwargs):
        noise_type = kwargs['noise_type']
        k = kwargs['k']
        noise_rate = kwargs['noise_rate']
        seed = kwargs['seed']
        G = kwargs['G']
        cnt = kwargs['cnt']
        max_iter = kwargs['max_iter']
        local = kwargs['local']
        loss = kwargs['loss']
        num_epoch = kwargs['num_epoch']
        min_similarity = kwargs['min_similarity']
        Tii_offset = kwargs['Tii_offset']
        method = kwargs['method']

        self.config = global_var.SimiArgs(noise_rate=noise_rate,
                                        noise_type=noise_type,
                                        Tii_offset=Tii_offset,
                                        k=k,
                                        G=G,
                                        seed=seed,
                                        cnt=cnt,
                                        max_iter=max_iter,
                                        local=local,
                                        loss=loss,
                                        num_epoch=num_epoch,
                                        min_similarity=min_similarity,
                                        method=method)
        self.config.device = self.model.device

        logging.debug("Running SIMIFEAT...")

        N = data_aq.data.shape[0]
        rand_inds = np.arange(N)
        np.random.shuffle(rand_inds)

        # Pass 1
        noisy_inds, test_inds = rand_inds[:N//2], rand_inds[N//2:]
        temp_data_aq = copy.deepcopy(data_aq)
        temp_data_aq.set_inds(noisy_inds)
        self.fit(temp_data_aq) # Train model on half the training data randomly chosen
        del temp_data_aq

        temp_data_aq = copy.deepcopy(data_aq)
        temp_data_aq.set_inds(test_inds)
        sel_inds_1 = self._noniterate_detect(temp_data_aq, desc='SimiFeat First Pass')
        logging.debug(f"First Pass num issues detected: {sel_inds_1.sum()}")

        # Pass 2
        temp_data_aq = copy.deepcopy(data_aq)
        temp_data_aq.set_inds(test_inds)
        self.fit(temp_data_aq) # Train model on half the training data randomly chosen
        del temp_data_aq

        temp_data_aq = copy.deepcopy(data_aq)
        temp_data_aq.set_inds(noisy_inds)
        sel_inds_2 = self._noniterate_detect(temp_data_aq, desc='SimiFeat Second Pass')
        logging.debug(f"Second Pass num issues detected: {sel_inds_2.sum()}")

        mask = np.array([False]*N)
        mask[test_inds[sel_inds_1].tolist() + noisy_inds[sel_inds_2].tolist()] = True

        # print(mask.sum())
        # raise KeyboardInterrupt
        return mask

    def fit(self, data_aq):
        return self.model.fit(data_aq,
                              lr_tune=True)