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
from transformers import AutoModel, AutoModelForSequenceClassification

SENTENCE_TRANSFORMERS = ["all-distilroberta-v1", "all-MiniLM-L6-v2"]

def getResnet18(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained, force_reload=force_reload)

def getResnet34(pretrained=True, force_reload=False):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained, force_reload=force_reload)

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