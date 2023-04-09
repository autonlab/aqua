import torch
from sklearn.ensemble import RandomForestClassifier

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLPNet, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, input_dim//2)
        self.layer2 = torch.nn.Linear(input_dim//2, input_dim//4)
        self.layer3 = torch.nn.Linear(input_dim//4, max(input_dim//6, 1))

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.dropout(self.relu(self.layer3(x)))

        return x


class TabularNet(torch.nn.Module):
    def __init__(self, model_type, output_dim, **kwargs):
        super(TabularNet, self).__init__()
        self.output_dim = output_dim
        self.model = self.__get_model(model_type, **kwargs)
        self.linear = torch.nn.Linear(max(kwargs['input_dim']//6, 1), self.output_dim)
        
    def __get_model(self, model_type, **kwargs):
        if model_type=='mlp':
            return MLPNet(kwargs['input_dim'])
        
    def forward(self, x, return_feats=False, **kwargs):
        feats = self.model(x)
        x = self.linear(feats)

        if not return_feats:
            return x 
        else:
            return x, feats