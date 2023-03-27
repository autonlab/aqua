import torch
from aqua.models.base_architectures.base_utils import *

class BertNet(torch.nn.Module):
    def __init__(self, model_type, output_dim):
        super(BertNet, self).__init__()
        self.model, final_dim = getBertModel(model_type)
        self.fc1 = torch.nn.Linear(final_dim, 100)
        self.fc2 = torch.nn.Linear(100, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x, **kwargs):
        attention_mask = kwargs['attention_mask'] if 'attention_mask' in kwargs else None
        return_feats = False if 'return_feats' not in kwargs else kwargs['return_feats']
        feats = self.model(input_ids=x,
                           attention_mask=attention_mask)[0][:,0]
        x = self.fc1(feats)
        x = self.relu(x)

        x = self.fc2(x)
        if not return_feats:
            return x
        else:
            return x, return_feats