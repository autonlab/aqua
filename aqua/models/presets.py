import torch
import numpy as np
from sklearn.base import BaseEstimator

from aqua.data import Aqdata, TestAqdata
from torch.utils.data import DataLoader

def getResnet18(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained)

def getResnet34(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained)



class ConvNet(BaseEstimator):
    def __init__(self, 
                model_type,
                epochs=6,
                batch_size=64,
                lr=0.01,
                momentum=0.5,
                seed=1,
                device='cpu'):
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = None
        if model_type == 'resnet34':
            self.model = getResnet34()
        elif model_type == 'resnet18':
            self.model = getResnet18()
        else:
            raise RuntimeError(f"Given model type: {model_type} is not supported")

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.device = device

        if not torch.cuda.is_available() and 'cuda' in device:
            device = 'cpu'
            print("Cuda not supported in a CPU only machine, defaulting to CPU device")

        self.device = torch.device(device)

        # Push model to device
        self.model = self.model.to(device)

    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }

    def fit(self, data, labels):
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

        for epoch in range(1, self.epochs+1):
            self.model.train()

            print("Running epoch: ", epoch)
            for batch_idx, (data, target, _) in enumerate(trainloader):
                data, target = torch.from_numpy(data), torch.from_numpy(target)
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                preds = self.model(data)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
    
    def predict_proba(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
