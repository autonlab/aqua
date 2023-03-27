import torch
from aqua.models.base_architectures.base_utils import *

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
        elif model_type == 'mobilenet_v2':
            return getMobilenetv2(), 1000
        else:
            raise RuntimeWarning(f"Given model type: {model_type} is not supported")
        
        return None, 0

    def forward(self, x, **kwargs):
        return_feats = False if 'return_feats' not in kwargs else kwargs['return_feats']
        feats = self.model(x)
        x = self.linear(feats)
        if not return_feats:
            return x
        else:
            return x, feats