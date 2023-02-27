import torch
import torch.nn as nn

import cifar.model as cifar
import mnist.model as mnist
import breeds.model as breeds

from typing import *

def init_model(dataset : str, num_classes : List[int], device : torch.device):
    '''
    Load the correct model for each dataset.
    '''
    if dataset == 'MNIST':
        model = nn.DataParallel(mnist.CNN(num_classes)).to(device)
    elif dataset == 'CIFAR':
        model = nn.DataParallel(cifar.ResNet18(num_classes)).to(device) 
    elif dataset == 'BREEDS':
        model = nn.DataParallel(breeds.ResNet18(num_classes)).to(device)
    return model