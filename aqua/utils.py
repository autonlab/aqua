import random, os
import numpy as np
import torch
import pydicom as dicom
from typing import Union

from aqua.configs import model_configs, data_configs, main_config




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
                                      momentum=model_configs['base'][architecture]['momentum'],
                                      weight_decay=model_configs['base'][architecture]['weight_decay'])
    else:
        optim = torch.optim.Adam(model.parameters(),
                                       lr=model_configs['base'][architecture]['lr'])
    return optim


def config_sanity_checks():
    """
    Confirms the validity of main_config.json
    """
    incorrect_datasets = []
    for dataset in main_config['datasets']:
        if dataset not in data_configs:
            incorrect_datasets.append(dataset)

    if len(incorrect_datasets) > 0:
        raise RuntimeError(f"Incorrect datasets provided in main_config.json: {incorrect_datasets}, currently supported datasets: {list(data_configs.keys())}")

    incorrect_cleaning_methods = []
    for method in main_config["methods"]:
        if method not in model_configs['cleaning']:
            incorrect_cleaning_methods.append(method)

    if len(incorrect_cleaning_methods) > 0:
        raise RuntimeError(f"Incorrect cleaning methods provided in main_config.json: {incorrect_cleaning_methods}, currently supported datasets: {list(model_configs['cleaning'].keys())}")

    incorrect_architectures = []
    for key, value in main_config['architecture'].items():
        if value not in model_configs['base']:
            incorrect_architectures.append(value)

    if len(incorrect_architectures) > 0:
        raise RuntimeError(f"Incorrect base architecture provided in main_config.json: {incorrect_architectures}, currently supported datasets: {list(model_configs['base'].keys())}")
    

def get_available_gpus():
    """
    Get list of all available, empty GPUs
    """
    avail_gpus = []
    for i in range(torch.cuda.device_count()):
        free_mem = torch.cuda.mem_get_info(i)[0] * 1e-9
        if free_mem > 10.0:
            avail_gpus.append(f'cuda:{i}') 
    return avail_gpus

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