import torch
from aqua.models.base_architectures.base_utils import *

from torchvision.transforms import RandomHorizontalFlip, ColorJitter

class ConvNet(torch.nn.Module):
    def __init__(self, model_type, output_dim, **kwargs):
        super(ConvNet, self).__init__()
        self.output_dim = output_dim
        self.model, final_dim = self._get_model(model_type)
        self.linear = torch.nn.Linear(final_dim, output_dim)
        self.final_dim = final_dim

        self.transforms = torch.nn.Sequential(RandomHorizontalFlip(0.5), ColorJitter(0.2, 0.2))

    def _get_model(self, model_type):
        if model_type == 'resnet34':
            return getResnet34(), 1000
        elif model_type == 'resnet18':
            return getResnet18(), 1000
        elif model_type == 'mobilenet_v2':
            return getMobilenetv2(), 1000
        else:
            raise RuntimeWarning(f"Given model type: {model_type} is not supported")

    def forward(self, x, kwargs={}, return_feats=False):
        #print(x.shape)
        # TODO : vedant : make this cleaner?
        if x.shape[-1] == 1:
            x, feats = torch.randn(x.shape[0], self.output_dim), torch.randn(x.shape[0], self.final_dim)
            if not return_feats: return x
            else: return x, feats
        #x = self.transforms(x)
        feats = self.model(x)
        x = self.linear(feats)
        if x.shape[0] != feats.shape[0]:
            x = x.unsqueeze(0)
        if not return_feats:
            return x
        else:
            return x, feats