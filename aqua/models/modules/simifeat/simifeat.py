import torch
import numpy as np

from .sim_utils import noniterate_detection
from . import global_var as global_var

from aqua.data import Aqdata
from torch.utils.data import DataLoader

class SimiFeat:
    def __init__(self, model, 
                 noise_type="instance",
                 k=10, 
                 noise_rate=0.4, 
                 seed=1,
                 G=10,
                 cnt=15000,
                 max_iter=400,
                 local=False,
                 loss='fw',
                 num_epoch=1,
                 min_similarity=0.0,
                 Tii_offset=1.0,
                 method='rank1'):
        
        self.model = model
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


    def find_label_issues(self, data, labels):
        num_classes = np.unique(labels).shape[0]
        N = labels.shape[0]
        self.config.cnt = min(self.config.cnt, data.shape[0]//4)

        self.config.num_classes = num_classes
        dataset = Aqdata(data, labels)
        trainloader = DataLoader(dataset,
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

            for i_batch, (feature, label, index, _) in enumerate(trainloader):
                feature, label = feature.to(self.model.device).float(), label.to(self.model.device)
                with torch.no_grad():    
                    _, extracted_feat = self.model.model(feature, return_feats=True)
                for i in range(extracted_feat.shape[0]):
                    record[label[i]].append({'feature':extracted_feat[i].detach().cpu(), 'index':index[i]})
                if i_batch > 200:
                    break

            if self.config.method == 'both':
                #rank1 + mv
                self.config.method = 'rank1'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())

                sel_clean_rec[epoch][np.array(sel_clean)] += 0.5
                sel_times_rec[np.array(sel_idx)] += 0.5

                self.config.method = 'mv'
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                sel_clean_rec[epoch][np.array(sel_idx)] += 0.5

                self.config.method = 'both'
                sel_times_rec[np.array(sel_idx)] += 0.5
            else:
                # use one method
                sel_noisy, sel_clean, sel_idx = noniterate_detection(self.config, record, dataset, 
                                                                    sel_noisy=sel_noisy_rec.copy())
                if self.config.num_epoch > 1:
                    sel_clean_rec[epoch][np.array(sel_clean)] = 1
                    sel_times_rec[np.array(sel_idx)] += 1

            aa = np.sum(sel_clean_rec[:epoch + 1], 0) / sel_times_rec
            nan_flag = np.isnan(aa)
            aa[nan_flag] = 0

            sel_clean_summary = np.round(aa).astype(bool)
            sel_noisy_summary = np.round(1.0 - aa).astype(bool)
            sel_noisy_summary[nan_flag] = False

            #print(np.sum(sel_noisy_summary))
            return sel_noisy_summary