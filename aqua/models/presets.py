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

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x



class ImageNet(BaseEstimator):
    def __init__(self, 
                model_type,
                output_dim,
                epochs=6,
                batch_size=64,
                lr=0.01,
                momentum=0.5,
                seed=1,
                device='cpu'):
        
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
        self.output_dim = output_dim

        # Push model to device
        self.model = ConvNet(model_type, output_dim).to(device)

    def get_params(self, deep=True):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "momentum": self.momentum,
            "device": self.device,
            "seed": self.seed,
            "model_type": self.model_type,
            "output_dim": self.output_dim
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(parameter, value)

        if 'model_type' in params:
            self.model = ConvNet(params['model_type'], params['output_dim']).to( params['device'])

        return self

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
                if batch_idx==0:
                    print(data.shape, target.shape)
                #data, target = torch.from_numpy(data), torch.from_numpy(target)
                data, target = data.float(), target.long()
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                preds = self.model(data)
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
    
    def predict_proba(self, data):
        dataset = TestAqdata(data)
        testloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=4)
        preds = []
        for batch_idx, data in enumerate(testloader):
            data = data.float().to(self.device)
            preds.append(self.model(data))

        return torch.vstack(preds).detach().cpu().numpy()

    def predict(self,data):
        raise NotImplementedError
