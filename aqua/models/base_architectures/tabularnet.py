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

from typing import Optional, List
import torch
from sklearn.ensemble import RandomForestClassifier

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim:int, layers:Optional[List]=None, p:float=0.2):
        super(MLPNet, self).__init__()
        if layers is None or len(layers) == 0: 
            layers = [input_dim//2, input_dim//4]    
        
        self.modules = [
            torch.nn.Linear(input_dim, layers[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=p),
            ] 
                
        for i in range(len(layers)-1):
            self.modules.extend([
                torch.nn.Linear(layers[i], layers[i+1]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=p)
                ]
            )
        
        self.model = torch.nn.Sequential(*self.modules)
    
    def forward(self, x):
        return self.model(x)

class TabularNet(torch.nn.Module):
    def __init__(self, model_type, output_dim, **kwargs):
        super(TabularNet, self).__init__()
        self.output_dim = output_dim
        self.model = self.__get_model(model_type, **kwargs)
        
        # Make the output layer
        if len(kwargs["layers"]) == 0 or kwargs["layers"] is None: 
            penultimate_layer_dim = kwargs['input_dim']//4
        else:
            penultimate_layer_dim = kwargs['layers'][-1]
        
        self.output_layer = torch.nn.Linear(penultimate_layer_dim, 
                                            self.output_dim)
        
    def __get_model(self, model_type, **kwargs):
        if model_type=='mlp':
            return MLPNet(kwargs['input_dim'], 
                          kwargs['layers'],
                          kwargs['p'])
        
    def forward(self, x, return_feats=False, **kwargs):
        feats = self.model(x)
        x = self.output_layer(feats)
        if x.shape[0] != feats.shape[0]:
            x = x.unsqueeze(0)
        if not return_feats:
            return x 
        else:
            return x, feats