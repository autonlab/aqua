import torch 
from transformers import AutoModel

def getResnet18(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained, force_reload=force_reload)

def getResnet34(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained, force_reload=force_reload)

def getMobilenetv2(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained, force_reload=force_reload)

def getBertModel(modelname):
    model =  AutoModel.from_pretrained(modelname)
    if modelname == 'roberta-base':
        return model, 768
    else:
        return model, None