import numpy as np
from torch.utils.data import DataLoader
from typing import *

from hierarchy.data import get_k_shot
import cifar.data as cifar
import mnist.data as mnist
import breeds.data as breeds


def make_kshot_loader(num_workers : int, batch_size : int, k : int, layer : str, 
                      task : str, seed : int, dataset : str, case : int, breeds_setting : str) -> Tuple[DataLoader, DataLoader]:
    '''
    Prepare one-shot train loader and full test loader. In train dataset, we have k
    image for each class (of indicated layer) in train set.
    Only call this function on unseen **target** train/test set.
    Args:
        batch_size : batch size of train/test loader
        k : k shot on train set
        layer : on which layer to sample k image per class
        task : in / sub
    '''
    # we use one shot setting only for target hierarchy
    # where we have to fine tune because of distribution shift at fine level
    if task == 'sub':
        train_dataloader, test_dataloader = make_dataloader(num_workers, batch_size, 'sub_split_downstream', dataset, case, breeds_setting)
    elif task == 'in':
        train_dataloader, test_dataloader = make_dataloader(num_workers, batch_size, 'in_split_downstream', dataset, case, breeds_setting)
    train_subset, _ = get_k_shot(k, train_dataloader.dataset, layer, seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                                pin_memory=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataloader.dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=False, num_workers=num_workers)
    dataloaders = (train_loader, test_loader)
    return dataloaders

def make_dataloader(num_workers : int, batch_size : int, task : str, dataset : str, 
                    case = None, breeds_setting : str = None) -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader.
        task : {sub/in}_split_{pretrain/downstream} | full | outlier
        dataset : MNIST | CIFAR | BREEDS
        case : case for MNIST. See mnist.make_dataloader.
        breeds_setting : living17 | entity13 | entity30 | nonliving26
    '''
    if dataset == 'MNIST':
        if case is None:
            raise ValueError("Please specify --case as any of 0/1")
        train_loader, test_loader = mnist.make_dataloader(num_workers, batch_size, task, case)
    elif dataset == 'CIFAR':
        train_loader, test_loader = cifar.make_dataloader(num_workers, batch_size, task)
    elif dataset == 'BREEDS':
        if breeds_setting is None:
            raise ValueError("Please specify --breeds_setting as any of living17 | entity13 | entity30 | nonliving26")
        train_loader, test_loader = breeds.make_dataloader(num_workers, batch_size, task, breeds_setting)
    return train_loader, test_loader
