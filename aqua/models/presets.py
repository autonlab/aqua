import torch, sys
import numpy as np
from sklearn.base import BaseEstimator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader
from aqua.utils import clear_memory

from transformers import AutoModel

import pdb
            



# class BaseNet(BaseEstimator):
#     def __init__(self):
#         # Relevent training metrics go here
#         self.train_metrics = None
#         self.reset_train_metrics()

#     def get_params(self):
#         raise NotImplementedError("Must be implemented by inheriting class")

#     def set_params(self):
#         raise NotImplementedError("Must be implemented by inheriting class")

#     def fit(self, *args):
#         raise NotImplementedError("Must be implemented by inheriting class")
    
#     def predict_proba(self, *args):
#         raise NotImplementedError("Must be implemented by inheriting class")

#     def predict(self, *args):
#         raise NotImplementedError("Must be implemented by inheriting class")

#     def get_training_metrics(self):
#         return self.train_metrics

#     def reset_train_metrics(self):
#         self.train_metrics = {
#             "epoch": [],
#             "batch" : [],
#             "output" : [],
#             "target" : [],
#             "sample_id" : []
#         }



# class ImageNet(BaseNet):
#     def __init__(self, 
#                 model_type,
#                 output_dim,
#                 epochs=6,
#                 batch_size=64,
#                 lr=0.01,
#                 momentum=0.5,
#                 seed=1,
#                 lr_drops = [0.5],
#                 device='cpu'):
        
#         super(ImageNet, self).__init__()

#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         if not torch.cuda.is_available() and 'cuda' in device:
#             device = 'cpu'
#             print("Cuda not supported in a CPU only machine, defaulting to CPU device")

#         self.device = torch.device(device)

#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.lr = lr
#         self.momentum = momentum
#         self.device = device
#         self.seed = seed
#         self.model_type = model_type
#         self.lr_drops = lr_drops
#         self.output_dim = output_dim

#         # Push model to device
#         self.model = ConvNet(model_type, output_dim).to(device)

#     def reinit_model(self, model_type, output_dim):
#         if self.model is not None:
#             del self.model

#         self.reset_train_metrics()
#         self.model = ConvNet(model_type, output_dim).to(self.device)
#         self.model_type = model_type
#         self.output_dim = output_dim

#     def get_params(self, deep=True):
#         return {
#             "batch_size": self.batch_size,
#             "epochs": self.epochs,
#             "lr": self.lr,
#             "momentum": self.momentum,
#             "device": self.device,
#             "seed": self.seed,
#             "model_type": self.model_type,
#             "output_dim": self.output_dim,
#             "lr_drops" : self.lr_drops
#         }

#     def set_params(self, **params):
#         for parameter, value in params.items():
#             setattr(parameter, value)

#         if 'model_type' in params:
#             #self.model = ConvNet(params['model_type'], params['output_dim']).to( params['device'])
#             self.reinit_model(params['model_type'], params['output_dim'])
#         return self
    
#     def _train_step(self, data, target, sample_ids, model, 
#                     optimizer, criterion, device):
#         data, target = data.float(), target.long()
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         preds = model(data)

#         # Save training metrics
#         self.train_metrics['output'].append(preds.cpu().detach())
#         self.train_metrics['target'].append(target.cpu().detach())
#         self.train_metrics['sample_id'].append(sample_ids.tolist())

#         loss = criterion(preds, target)
#         loss.backward()
#         optimizer.step()

#         del data
#         del target
#         del loss
#         del preds

#     def fit(self, *args, 
#             lr_tune=False,
#             early_stop=False,
#             data_kwargs={}):
#         """
#         Please refer to: https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py
#         """
#         if isinstance(args[0], Aqdata):
#             data_aq = args[0]
#         else:
#             data_aq = Aqdata(args[0], args[1], **data_kwargs)
#         trainloader = DataLoader(data_aq,
#                                  batch_size=self.batch_size,
#                                  shuffle=True,
#                                  num_workers=4)
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
#         criterion = torch.nn.CrossEntropyLoss()
#         scheduler = None

#         if lr_tune:
#             milestones = [int(lr_drop * self.epochs) for lr_drop in (self.lr_drops or [])]
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                             milestones=milestones,
#                                                             gamma=0.1)

#         for epoch in range(1, self.epochs+1):
#             self.model.train()

#             print("Running epoch: ", epoch)
#             for batch_idx, (data, target, idx, _, kwargs) in enumerate(trainloader):
#                 self._train_step(data, target,
#                                 idx, self.model, 
#                                 optimizer, criterion, 
#                                 self.device)

#             if scheduler:
#                 scheduler.step()
#                 if early_stop and (scheduler.get_last_lr()[-1] < self.lr):
#                     break
    
#     def predict_proba(self, data_aq):
#         if isinstance(data_aq, TestAqdata):
#             testloader = DataLoader(data_aq,
#                                     batch_size=self.batch_size,
#                                     num_workers=4)
#             preds = []
#             self.model.eval()
#             for batch_idx, (data, idx, data_kwargs) in enumerate(testloader):
#                 data = data.float().to(self.device)
#                 preds.append(self.model(data, **data_kwargs).detach().cpu())
#                 del data

#             return torch.nn.Softmax(dim=1)(torch.vstack(preds)).numpy()
#         elif isinstance(data_aq, Aqdata):
#             testloader = DataLoader(data_aq,
#                                     batch_size=self.batch_size,
#                                     num_workers=4)
#             preds = []
#             self.model.eval()
#             for batch_idx, (data, _, idx, _, data_kwargs) in enumerate(testloader):
#                 data = data.float().to(self.device)
#                 preds.append(self.model(data, **data_kwargs).detach().cpu())
#                 del data

#             return torch.nn.Softmax(dim=1)(torch.vstack(preds)).numpy()
#         else:
#             self.model.eval()
#             data = torch.from_numpy(data_aq)
#             preds = []
#             if data_aq.shape[0] > 1:
#                 for i in range(0, data.shape[0], self.batch_size): 
#                     x = data[i:i+self.batch_size].float().to(self.device)
#                     preds.append(self.model(x).detach().cpu().numpy())
#                     del x
#                 preds = torch.nn.Softmax(dim=1)(torch.from_numpy(np.concatenate(preds))).numpy()
#             else:
#                 preds = torch.nn.Softmax(dim=1)(self.model(torch.from_numpy(data_aq).float().to(self.device))).detach().cpu().numpy()
#             return preds

#     def predict(self, data):
#         self.model.eval()
#         probs = self.predict_proba(data)
#         return np.argmax(probs, axis=1)
    



class AqNet:
    def __init__(self, 
                model,
                output_dim,
                epochs=6,
                batch_size=64,
                lr=0.01,
                lr_drops = [0.5],
                device='cpu'):
        
        #super(TextNet, self).__init__()

        self.train_metrics = None
        self.reset_train_metrics()

        if not torch.cuda.is_available() and 'cuda' in device:
            device = 'cpu'
            print("Cuda not supported in a CPU only machine, defaulting to CPU device")

        self.device = torch.device(device)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.lr_drops = lr_drops
        self.output_dim = output_dim

        # Push model to device
        self.model = model.to(device)

    def reset_train_metrics(self):
        self.train_metrics = {
            "epoch": [],
            "batch" : [],
            "output" : [],
            "target" : [],
            "sample_id" : []
        }

    def reinit_model(self, model, output_dim):
        if self.model is not None:
            del self.model
        self.reset_train_metrics()
        self.model = model.to(self.device)
        self.output_dim = output_dim

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
    
    def __train_step(self, data, target, sample_ids, data_kwargs, model, 
                    optimizer, criterion, device):
        data, target = data.long(), target.long()
        data, target = data.to(device), target.to(device)
        data_kwargs = self.__move_data_kwargs(data_kwargs, device)

        optimizer.zero_grad()
        preds = model(data, data_kwargs)

        # Save training metrics
        self.train_metrics['output'].append(preds.cpu().detach())
        self.train_metrics['target'].append(target.cpu().detach())
        self.train_metrics['sample_id'].append(sample_ids.tolist())

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        
        del preds
        del data 
        del target
        del loss 
        del data_kwargs

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
        trainloader = DataLoader(data_aq,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = None

        if lr_tune:
            milestones = [int(lr_drop * self.epochs) for lr_drop in (self.lr_drops or [])]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=milestones,
                                                            gamma=0.1)

        for epoch in range(1, self.epochs+1):
            self.model.train()

            print("Running epoch: ", epoch)
            for batch_idx, (data, target, idx, _, data_kwargs) in enumerate(trainloader):
                self.__train_step(data, target, idx,
                                data_kwargs, self.model, 
                                optimizer, criterion, 
                                self.device)

            if scheduler:
                scheduler.step()
                if early_stop and (scheduler.get_last_lr()[-1] < self.lr):
                    break

        del optimizer 
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
            for batch_idx, (data, idx, data_kwargs) in enumerate(testloader):
                data = data.long().to(self.device)
                preds.append(self.model(data))

            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).detach().cpu().numpy()
        elif isinstance(args[0], Aqdata):
            testloader = DataLoader(args[0],
                                    batch_size=self.batch_size,
                                    num_workers=4)
            preds = []
            self.model.eval()
            for batch_idx, (data, _, idx, _, data_kwargs) in enumerate(testloader):
                data, attention_mask = data.long().to(self.device), attention_mask.long().to(self.device)
                preds.append(self.model(data))
            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).detach().cpu().numpy()
        else:
            data_aq = args[0]
            return torch.nn.Softmax(dim=1)(self.model(torch.from_numpy(data_aq).long().to(self.device))).detach().cpu().numpy()

    def predict(self, *args,
                data_kwargs={}):
        self.model.eval()
        probs = self.predict_proba(*args, data_kwargs=data_kwargs)
        return np.argmax(probs, axis=1)
