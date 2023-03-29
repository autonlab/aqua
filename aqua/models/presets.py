import torch, sys
import numpy as np
from sklearn.base import BaseEstimator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader
from aqua.utils import clear_memory

from transformers import AutoModel



class AqNet(BaseEstimator):
    def __init__(self, 
                model,
                output_dim,
                epochs=6,
                batch_size=64,
                lr=0.01,
                lr_drops = [0.5],
                device='cpu',
                optimizer=None):
        
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
        data, target = data.to(self.device), target.to(self.device)
        data_kwargs = self.__move_data_kwargs(data_kwargs, self.device)

        self.optimizer.zero_grad()
        preds = self.model(data, data_kwargs)

        # Save training metrics
        self.train_metrics['output'].append(preds.cpu().detach())
        self.train_metrics['target'].append(target.cpu().detach())
        self.train_metrics['sample_id'].append(sample_ids.tolist())

        loss = criterion(preds, target)
        loss.backward()
        self.optimizer.step()
        
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
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = None

        if lr_tune:
            milestones = [int(lr_drop * self.epochs) for lr_drop in (self.lr_drops or [])]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=0.1)

        for epoch in range(1, self.epochs+1):
            self.model.train()

            #print("Running epoch: ", epoch)
            for batch_idx, (data, target, idx, _, data_kwargs) in enumerate(trainloader):
                self.__train_step(data, target, idx,
                                data_kwargs,
                                criterion)

            if scheduler:
                scheduler.step()
                if early_stop and (scheduler.get_last_lr()[-1] < self.lr):
                    break
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
            for batch_idx, (data, _, _) in enumerate(testloader):
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
            for batch_idx, (data, _, _, _, _) in enumerate(testloader):
                data = data.to(self.device)
                preds.append(self.model(data).detach().cpu())
                del data

            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).numpy()
        else:
            self.model.eval()
            data_aq = args[0]
            data = torch.from_numpy(data_aq)
            preds = []
            if data_aq.shape[0] > 1:
                for i in range(0, data.shape[0], self.batch_size): 
                    x = data[i:i+self.batch_size].float().to(self.device)
                    preds.append(self.model(x).detach().cpu().numpy())
                    del x
                preds = torch.nn.Softmax(dim=1)(torch.from_numpy(np.concatenate(preds))).numpy()
            else:
                preds = torch.nn.Softmax(dim=1)(self.model(torch.from_numpy(data_aq).float().to(self.device))).detach().cpu().numpy()
            return preds

    def predict(self, *args,
                data_kwargs={}):
        self.model.eval()
        probs = self.predict_proba(*args, data_kwargs=data_kwargs)
        return np.argmax(probs, axis=1)
