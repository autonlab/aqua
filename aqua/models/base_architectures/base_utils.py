import torch 
from transformers import AutoModel, AutoModelForSequenceClassification

SENTENCE_TRANSFORMERS = ["all-distilroberta-v1", "all-MiniLM-L6-v2"]

def getResnet18(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained, force_reload=force_reload)

def getResnet34(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained, force_reload=force_reload)

def getMobilenetv2(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained, force_reload=force_reload)

def getBertModel(modelname):
    if modelname in SENTENCE_TRANSFORMERS:
        modelname = f"sentence-transformers/{modelname}"
    model = AutoModel.from_pretrained(modelname)
    if modelname == 'sentence-transformers/all-MiniLM-L6-v2':
        return model, 384
    elif modelname == 'sentence-transformers/all-distilroberta-v1':
        return model, 768
    else:
        return model, None