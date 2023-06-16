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

import pdb

class BertNet(torch.nn.Module):
    def __init__(self, model_type, output_dim, **kwargs):
        super(BertNet, self).__init__()
        self.model, final_dim = getBertModel(model_type)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # Freeze BERT except the pooler layer
        # for name, param in self.model.named_parameters():
        #     if not name.startswith('pooler'):
        #         print("pool frozen")
        #         param.requires_grad = False
        self.fc1 = torch.nn.Linear(final_dim, 768)
        self.fc2 = torch.nn.Linear(768, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x, kwargs={}, return_feats=False):
        attention_mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else None
        x = x.int()
        feats = self.model(input_ids=x,
                           attention_mask=attention_mask).pooler_output #[0][:,0]
        #pdb.set_trace()
        x = self.fc1(feats)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        if x.shape[0] != feats.shape[0]:
            x = x.unsqueeze(0)
        if not return_feats:
            return x
        else:
            return x, feats