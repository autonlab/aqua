import torch
from sklearn.ensemble import RandomForestClassifier

class MLPNet(torch.nn.Module):
    def __init__(self, output_dim, input_dim=None):
        layer1 = torch.nn.Linear(input_dim, output_dim*75)
        relu = torch.nn.ReLU()
        dropout = torch.nn.Dropout(p=0.2)

class TabularNet:
    def __init__(self, model_type, output_dim, **kwargs):
        self.model = model_type(model_type)
        
    def _get_model(self, model_type):
        if model_type=='random_forest':
            return RandomForestClassifier()