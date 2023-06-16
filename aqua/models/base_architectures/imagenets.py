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