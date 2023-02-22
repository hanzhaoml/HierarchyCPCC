import os
import random

import torch
from torch import Tensor
import numpy as np


def seed_everything(seed : int) -> None: 
    '''
        Seed everything for reproducibility.
        Args:
            seed : any integer
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_layer_prob_from_fine(prob_fine : Tensor, layer_fine_map : np.ndarray) -> Tensor:
    '''
        Marginalize non-fine layer probability from fine level output.
        Args:
            prob_fine : batch_size * number of fine classes
            layer_fine_map : 1d array of length = number of fine classes,
            where L[i] is layer label for fine label i.
    '''
    prob_layer = []
    num_layer_targets = len(np.unique(layer_fine_map))
    device = prob_fine.device
    for cls in range(num_layer_targets):
        prob_layer.append(torch.sum(torch.index_select(prob_fine,1,torch.tensor((layer_fine_map == cls).nonzero()[0],device=device)),1,keepdim=True))
    prob_layer = torch.cat(prob_layer, dim=1)
    return prob_layer
