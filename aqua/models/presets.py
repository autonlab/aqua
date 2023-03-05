import torch
import numpy as np
from sklearn.base import BaseEstimator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader

def getResnet18(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained)

def getResnet34(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained)


class ConvNet(torch.nn.Module):
    def __init__(self, model_type, output_dim):
        super(ConvNet, self).__init__()
        self.model, final_dim = self._get_model(model_type)
        self.linear = torch.nn.Linear(final_dim, output_dim)

    def _get_model(self, model_type):
        if model_type == 'resnet34':
            return getResnet34(), 1000
        elif model_type == 'resnet18':
            return getResnet18(), 1000
        else:
            raise RuntimeWarning(f"Given model type: {model_type} is not supported")
        
        return None, 0

    def forward(self, x, return_feats=False):
        feats = self.model(x)
        x = self.linear(feats)
        if not return_feats:
            return x
        else:
            return x, feats


class BaseNet(BaseEstimator):
    def __init__(self):
        # Relevent training metrics go here
        self.train_metrics = None
        self.reset_train_metrics()

    def get_params(self):
        raise NotImplementedError("Must be implemented by inheriting class")

    def set_params(self):
        raise NotImplementedError("Must be implemented by inheriting class")

    def fit(self, *args):
        raise NotImplementedError("Must be implemented by inheriting class")
    
    def predict_proba(self, *args):
        raise NotImplementedError("Must be implemented by inheriting class")

    def predict(self, *args):
        raise NotImplementedError("Must be implemented by inheriting class")

    def _train_step(self, data, target, sample_ids, model, 
                    optimizer, criterion, device):
        data, target = data.float(), target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        preds = model(data)

        # Save training metrics
        self.train_metrics['output'].append(preds.cpu().detach())
        self.train_metrics['target'].append(target.cpu().detach())
        self.train_metrics['sample_id'].append(sample_ids.tolist())

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

    def get_training_metrics(self):
        return self.train_metrics

    def reset_train_metrics(self):
        self.train_metrics = {
            "epoch": [],
            "batch" : [],
            "output" : [],
            "target" : [],
            "sample_id" : []
        }



class ImageNet(BaseNet):
    def __init__(self, 
                model_type,
                output_dim,
                epochs=6,
                batch_size=64,
                lr=0.01,
                momentum=0.5,
                seed=1,
                lr_drops = [0.5],
                device='cpu'):
        
        super(ImageNet, self).__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)

        if not torch.cuda.is_available() and 'cuda' in device:
            device = 'cpu'
            print("Cuda not supported in a CPU only machine, defaulting to CPU device")

        self.device = torch.device(device)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.seed = seed
        self.model_type = model_type
        self.lr_drops = lr_drops
        self.output_dim = output_dim

        # Push model to device
        self.model = ConvNet(model_type, output_dim).to(device)

    def reinit_model(self, model_type, output_dim):
        if self.model is not None:
            del self.model

        self.reset_train_metrics()
        self.model = ConvNet(model_type, output_dim).to(self.device)
        self.model_type = model_type
        self.output_dim = output_dim

    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "momentum": self.momentum,
            "device": self.device,
            "seed": self.seed,
            "model_type": self.model_type,
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

    def fit(self, data, labels, 
            lr_tune=False,
            early_stop=False):
        """
        Please refer to: https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py
        """
        dataset = Aqdata(data, labels)
        trainloader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=4)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
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
            for batch_idx, (data, target, idx, _) in enumerate(trainloader):
                self._train_step(data, target, 
                                idx, self.model, 
                                optimizer, criterion, 
                                self.device)

            if scheduler:
                scheduler.step()
                if early_stop and (scheduler.get_last_lr()[-1] < self.lr):
                    break
    
    def predict_proba(self, data):
        if data.shape[0] != 1:
            dataset = TestAqdata(data)
            testloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    num_workers=4)
            preds = []
            self.model.eval()
            for batch_idx, (data, idx) in enumerate(testloader):
                data = data.float().to(self.device)
                preds.append(self.model(data))

            return torch.nn.Softmax(dim=1)(torch.vstack(preds)).detach().cpu().numpy()
        else:
            return torch.nn.Softmax(dim=1)(self.model(torch.from_numpy(data).float().to(self.device))).detach().cpu().numpy()

    def predict(self, data):
        self.model.eval()
        probs = self.predict_proba(data)
        return np.argmax(probs, axis=1)
