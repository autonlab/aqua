import random, os
import numpy as np
import torch
import pydicom as dicom
from typing import Union

from aqua.configs import model_configs




def seed_everything(seed:int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clear_memory(*args):
    for arg in args:
        del arg

def get_optimizer(model:torch.nn.Module, 
                  architecture:str):
    """
    Initialize a new optimizer for a new model
    """
    if 'momentum' in model_configs['base'][architecture]:
        optim = torch.optim.SGD(model.parameters(),
                                      lr=model_configs['base'][architecture]['lr'],
                                      momentum=model_configs['base'][architecture]['momentum'])
    else:
        optim = torch.optim.Adam(model.parameters(),
                                       lr=model_configs['base'][architecture]['lr'])
    return optim

###################### DATA LOADING UTILS ####################
def __load_dcm(path):
    arr = np.repeat(dicom.dcmread(path).pixel_array[np.newaxis, :, :], 3, axis=0)
    arr = (arr - arr.min())/(arr.max() - arr.min())
    #arr -= [0.485, 0.456, 0.406]
    #arr /= [0.299, 0.224, 0.225]
    return arr.astype(np.float32)

def load_single_datapoint(path: str):
    if path.endswith('.dcm'):
        return __load_dcm(path)
    
def load_batch_datapoints(paths: Union[str, np.ndarray]):
    arrs = []
    for i in range(len(paths)):
        arrs.append(load_single_datapoint(paths[i]))
    return np.array(arrs)