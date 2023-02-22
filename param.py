import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import json

def init_optim_schedule(model : nn.Module, params : dict):
    optim_param, schedule_param = params['optimizer'], params['scheduler']
    optimizer = SGD(model.parameters(), **optim_param)
    scheduler = StepLR(optimizer, **schedule_param)
    return optimizer, scheduler

def load_params(dataset_name : str, task : str, layer : str = None, 
                breeds_setting : str = None) -> dict:
    '''
        task : pre / down
        breeds_setting : living17/nonliving26/entity13/entity30
    '''
    reset = {}
    if dataset_name == 'BREEDS':
        with open(f'./{dataset_name.lower()}/{breeds_setting}/{task}.json','r') as fp:
            params = json.load(fp)
    else:
        with open(f'./{dataset_name.lower()}/{task}.json', 'r') as fp:
            params = json.load(fp)
    if layer is not None:
        reset['epochs'] = params[layer]['epochs']
        reset['optimizer'] = params[layer]['optimizer']
        reset['scheduler'] = params[layer]['scheduler']
    else:
        reset['epochs'] = params['epochs']
        reset['optimizer'] = params['optimizer']
        reset['scheduler'] = params['scheduler']
    return reset