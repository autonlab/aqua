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

import torch, sys, logging
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader
from aqua.utils import clear_memory, load_batch_datapoints
import warnings


class AqNet(BaseEstimator):
    def __init__(self, 
                model,
                output_dim,
                epochs=6,
                batch_size=64,
                lr=0.01,
                lr_drops = [0.5],
                weighted_loss = False,
                device='cpu',
                optimizer=None):
        
        #super(TextNet, self).__init__()

        self.train_metrics = None
        self.reset_train_metrics()

        if not torch.cuda.is_available() and 'cuda' in device:
            device = 'cpu'
            warnings.warn("Cuda not supported in a CPU only machine, defaulting to CPU device", RuntimeWarning)

        self.device = torch.device(device)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.lr_drops = lr_drops
        self.output_dim = output_dim
        self.weighted_loss = weighted_loss
        self.data_loaded_dynamically = False  # Keeps track if data was loaded dynamically during `fit` : required for `predict_proba`

        # Push model to device
        self.model = model.to(device)
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def reset_train_metrics(self):
        self.train_metrics = {
            "epoch": [],
            "batch" : [],
            "output" : [],
            "target" : [],
            "sample_id" : []
        }

    def get_training_metrics(self):
        return self.train_metrics

    def reinit_model(self, model, optimizer):
        if self.model is not None:
            del self.model
            del self.optimizer
        self.reset_train_metrics()
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "device": self.device,
            "model": self.model,
            "output_dim": self.output_dim,
            "lr_drops" : self.lr_drops
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(parameter, value)

        if 'model_type' in params:
            #self.model = ConvNet(params['model_type'], params['output_dim']).to( params['device'])
            self.reinit_model(params['model_type'], params['output_dim'])
        return self
    
    def __move_data_kwargs(self, data_kwargs, device):
        for key, values in data_kwargs.items():
            values = values.to(device)
            if key == 'attention_mask':
                values = values.long()
            data_kwargs[key] = values.to(device)
        return data_kwargs
    
    def __train_step(self, data, target, sample_ids, data_kwargs, 
                    criterion):
        #print(data.shape[0], target.shape[0])
        data, target = data.to(self.device), target.to(self.device)
        data_kwargs = self.__move_data_kwargs(data_kwargs, self.device)

        self.optimizer.zero_grad()
        preds = self.model(data, data_kwargs)

        # Save training metrics
        self.train_metrics['output'].append(preds.cpu().detach().half().float())
        self.train_metrics['target'].append(target.cpu().detach())
        self.train_metrics['sample_id'].append(sample_ids.tolist())

        loss = criterion(preds, target)
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()
        
        del preds
        del data 
        del target
        del loss 
        del data_kwargs

        return loss_val

    def fit(self, *args, 
            lr_tune=False,
            early_stop=False,
            data_kwargs={}):
        """
        Please refer to: https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py
        """
        if isinstance(args[0], Aqdata):
            data_aq = args[0]
        else:
            data_aq = Aqdata(args[0], args[1], **data_kwargs)

        self.data_loaded_dynamically = data_aq.lazy_load
        loader = DataLoader(data_aq,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4)
        
        weights = None
        if self.weighted_loss:
            weights = 1 - (np.unique(data_aq.labels, return_counts=True)[1]/data_aq.labels.shape[0])
            weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        scheduler = None

        # if lr_tune:
        #     milestones = [int(lr_drop * self.epochs) for lr_drop in (self.lr_drops or [])]
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                     milestones=milestones,
        #                                                     gamma=0.1)
        # else:

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               patience=4,
                                                               factor=0.1,
                                                               min_lr=1e-6)


        for epoch in range(1, self.epochs+1):
            self.model.train()

            trainloader = tqdm(loader)
            trainloader.set_description(f"Epoch: {epoch}/{self.epochs}")
            avg_loss, loss_count = 0, 0
            prev_lr = self.optimizer.param_groups[0]['lr']
            for batch_idx, (data, target, idx, _, data_kwargs) in enumerate(trainloader):
                loss = self.__train_step(data, target, idx,
                                            data_kwargs,
                                            criterion)
                
                avg_loss += loss
                loss_count += 1
                res = {'loss': f"{loss:.3f} ({(avg_loss/loss_count):.3f})"}
                trainloader.set_postfix(**res)
                
                if batch_idx % 500 == 0:
                    logging.info(f"Epoch: {epoch}, Batch: {batch_idx}, Avg Loss: {res['loss']}")

            if scheduler:
                scheduler.step(avg_loss/loss_count)
                if self.optimizer.param_groups[0]['lr'] < prev_lr:
                    print("\n\nLR reduced on Plateu\n\n")
                # if early_stop and (scheduler.get_last_lr()[-1] > self.lr):
                #     logging.info("Model has early stopped!")
                #     break
            
            prev_lr = self.optimizer.param_groups[0]['lr']
            logging.info("\n\n")
        del scheduler
        del criterion
    
    def predict_proba(self, *args, 
                     data_kwargs={}):
        if isinstance(args[0], TestAqdata):
            testloader = DataLoader(args[0],
                                    batch_size=self.batch_size,
                                    num_workers=4)
            preds = []
            self.model.eval()
            for batch_idx, (data, _, _) in tqdm(enumerate(testloader), desc='Predicting probs'):
                data = data.to(self.device)
                preds.append(self.model(data).detach().cpu())
                del data

            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).numpy()
        elif isinstance(args[0], Aqdata):
            testloader = DataLoader(args[0],
                                    batch_size=self.batch_size,
                                    num_workers=4)
            preds = []
            self.model.eval()
            for batch_idx, (data, _, _, _, _) in tqdm(enumerate(testloader), desc='Predicting probs'):
                data = data.to(self.device)
                preds.append(self.model(data).detach().cpu())
                del data

            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).numpy()
        else:
            self.model.eval()
            data_aq = args[0]
            preds = []
            if data_aq.shape[0] > 1:
                for i in tqdm(range(0, data_aq.shape[0], self.batch_size), desc='Predicting probs'): 
                    x = data_aq[i:i+self.batch_size]
                    if self.data_loaded_dynamically:
                        x = load_batch_datapoints(x)
                    x = torch.from_numpy(x).to(self.device)
                    preds.append(self.model(x).detach().cpu().numpy())
                    del x
                preds = torch.nn.Softmax(dim=1)(torch.from_numpy(np.concatenate(preds))).numpy()
            else:
                if self.data_loaded_dynamically:
                    data_aq = load_batch_datapoints(data_aq)
                preds = torch.nn.Softmax(dim=1)(self.model(torch.from_numpy(data_aq).float().to(self.device))).detach().cpu().numpy()
            logging.debug(f"Pred shape after predict_proba {preds.shape}")
            return preds

    def predict(self, *args,
                data_kwargs={}):
        self.model.eval()
        probs = self.predict_proba(*args, data_kwargs=data_kwargs)
        return np.argmax(probs, axis=1)
