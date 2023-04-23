import os, shutil, copy, torch
import numpy as np
import pandas as pd
from aqua.configs import main_config

# AUM imports
from aum import AUMCalculator


class AUM:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.dthr = None
        self.orig_dim = self.model.output_dim


    def _fit_get_aum(self, thr_inds, iter='train'):
        # self._aum_calculator = AUMCalculator(os.getcwd())
        self._aum_calculator = AUMCalculator(os.path.join(main_config['results_dir'], 'results'))
        train_metrics = self.model.get_training_metrics()
        for i in range(len(train_metrics['output'])):
            self._aum_calculator.update(logits=train_metrics['output'][i],
                                        targets=train_metrics['target'][i],
                                        sample_ids=train_metrics['sample_id'][i])
        self._aum_calculator.finalize()
        aum_results_filepath = os.path.join(main_config['results_dir'], f'results/aum_values_{iter}.csv')
        shutil.move(os.path.join(main_config['results_dir'], 'results/aum_values.csv'), aum_results_filepath)
        aum_file = pd.read_csv(aum_results_filepath)
        aum_tensor = torch.tensor(aum_file["aum"].to_list())
        aum_wtr = torch.lt(aum_tensor.view(-1, 1),
                           aum_tensor[thr_inds].view(1, -1),
                           ).float().mean(dim=-1).gt(0.01).float()

        thresh = np.percentile(aum_file.iloc[thr_inds]['aum'].values, self.alpha*100)
        mask = np.array([True]*aum_file.shape[0])  # Selects train indices only, discards THR indices
        mask[thr_inds] = False

        # import pdb
        # pdb.set_trace()

        return np.array(aum_file.index)[mask][aum_file['aum'].values[mask] < thresh]
        

    def find_label_issues(self, data_aq,
                          **kwargs):
        # Read: https://discuss.pytorch.org/t/does-deepcopying-optimizer-of-one-model-works-across-the-model-or-should-i-create-new-optimizer-every-time/14359/6
        # Save initial states of models before training 
        orig_model = copy.deepcopy(self.model.model)
        orig_optim = type(self.optimizer)(orig_model.parameters(), lr=self.optimizer.defaults['lr'])
        orig_optim.load_state_dict(self.optimizer.state_dict())

        self.alpha = kwargs['alpha']
        # Refer to paper for training strategy

        labels = data_aq.labels
        # Randomly assign N/(c+1) data as the (c+1)th class
        label_val = np.unique(labels).max()+1
        N, c = labels.shape[0], np.unique(labels).shape[0]+1
        
        rand_inds = np.arange(N)
        np.random.shuffle(rand_inds)

        # Pass 1
        temp_data_aq = copy.deepcopy(data_aq)
        labels_step_1 = labels.copy()
        labels_step_1[rand_inds[:(N//c)]] = label_val
        temp_data_aq.labels = labels_step_1
        self.fit(temp_data_aq)
        incorrect_labels_idx = self._fit_get_aum(rand_inds[:(N//c)])
        del temp_data_aq
        #print(incorrect_labels_idx.shape)

        # Pass 2
        temp_data_aq = copy.deepcopy(data_aq)
        self.model.reinit_model(orig_model, orig_optim)
        labels_step_2 = labels.copy()
        labels_step_2[rand_inds[(N//c):]] = label_val
        temp_data_aq.labels = labels_step_2
        self.fit(temp_data_aq)
        incorrect_labels_idx_thresh = self._fit_get_aum(rand_inds[(N//c):], 'test')
        total_incorrect_labels = np.union1d(incorrect_labels_idx, incorrect_labels_idx_thresh)
        del temp_data_aq
        #print(incorrect_labels_idx_thresh.shape)
        #print(total_incorrect_labels.shape)

        mask = np.array([False]*labels.shape[0])  # Selects train indices only, discards THR indices
        mask[total_incorrect_labels] = True

        # Re-instantiate the model with correct number of output neurons
        return mask
    
    
    def fit(self, data_aq):        
        return self.model.fit(data_aq, 
                              lr_tune=True,
                              early_stop=True)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)