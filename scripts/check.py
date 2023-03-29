import torch, sys, random, numpy as np, os
sys.path.append('../')

from aqua.utils import seed_everything

import pdb

#seed_everything(42)

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Conv1d(3, 10, 3)
        self.layer2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))

def getResnet18(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained, force_reload=True)

def getResnet34(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained, force_reload=True)

def getMobilenetv2(pretrained=True):
    return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained, force_reload=True)

model_1 = getResnet34().to('cuda:3')
model_2 = getResnet34().to('cuda:3')
#model_1 = ConvNet().to('cuda:3')
#model_2 = ConvNet().to('cuda:3')

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

compare_models(model_1, model_2)

pdb.set_trace()