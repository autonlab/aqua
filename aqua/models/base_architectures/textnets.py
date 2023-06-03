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