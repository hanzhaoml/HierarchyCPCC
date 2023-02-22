from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from hierarchy.data import Hierarchy, get_k_shot
import numpy as np
from PIL import Image
from typing import *

class HierarchyMNIST(MNIST, Hierarchy):
    '''
    odd/even for coarse split
    type: if case == 0, odd/even split for coarse, <> 5 for mid.
    if case == 1, <> 5 for coarse, odd/even split for mid
    '''
    def __init__(self, case, *args, **kw):
        super(HierarchyMNIST, self).__init__(*args, **kw)
        self.case = case
        self.fine_map = np.array(range(10))
        self.fine_names = ["0","1","2","3","4","5","6","7","8","9"]

        if case == 0: # even/odd coarse split
            self.coarse_map = np.array([0,1,0,1,0,1,0,1,0,1])
            self.coarse_names = ["even","odd"]
        elif case == 1: # greater(>= 5)/less(<5) coarse split
            self.coarse_map = np.array([1,1,1,1,1,0,0,0,0,0])
            self.coarse_names = ["greater","less"]
        else:
            raise ValueError("Please specifiy --case")
        
        self.mid_map = np.array([1,3,1,3,1,3,0,2,0,2])
        self.mid_names = ["even_greater","even_less","odd_greater","odd_less"]
        
        self.img_size = 28
        self.channel = 1
        self.targets = [int(i) for i in self.targets]

        self.coarsest_map = np.empty(0)
        self.coarsest_names = []

    def __getitem__(self, index: int):
        img, target_fine = self.data[index], int(self.targets[index])
        target_mid = int(self.mid_map[target_fine])
        target_coarse = int(self.coarse_map[target_fine])
        target_coarsest = 0 # dummy
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_fine = self.target_transform(target_fine)

        return img, target_coarsest, target_coarse, target_mid, target_fine

class HierarchyMNISTSubset(HierarchyMNIST):
    def __init__(self, case, indices, fine_classes, *args, **kw):
        super(HierarchyMNISTSubset, self).__init__(case, *args, **kw)
        # don't touch coarsest labels

        self.data = self.data[indices]
        
        old_targets = list(np.array(self.targets)[indices]) # old fine targets, sliced
        fine_classes = np.array(sorted(fine_classes)) 
        self.fine_names = [self.fine_names[i] for i in fine_classes] # old fine names, sliced
        self.fine_map = np.array(range(len(fine_classes))) # number of fine classes subset
        self.targets = [list(fine_classes).index(i) for i in old_targets] # reset fine target id from 0

        # reset other hierarchy level index, from 0
        old_coarse_map = np.array([self.coarse_map[i] for i in fine_classes]) # subset of coarse fine map
        coarse_classes = np.unique(old_coarse_map)
        self.coarse_names = [self.coarse_names[cls] for cls in coarse_classes]
        self.coarse_map = np.array([list(coarse_classes).index(i) for i in old_coarse_map]) # argsort

        old_mid_map = np.array([self.mid_map[i] for i in fine_classes])
        mid_classes = np.unique(old_mid_map)
        self.mid_names = [self.mid_names[cls] for cls in mid_classes]
        self.mid_map = np.array([list(mid_classes).index(i) for i in old_mid_map])
    
    def __len__(self):
        return len(self.targets)

def make_dataloader(num_workers : int, batch_size : int, task : str, case : int) -> Tuple[DataLoader, DataLoader]:
    '''
    Args:
        case : 0 => coarse split by odd/even | 1 => coarse split by digit value relative to 5
    '''
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        train_dataset = HierarchyMNIST(case, root = './data',
                                    train = True,                         
                                    transform = transforms.ToTensor(),           
                                    )
        test_dataset = HierarchyMNIST(case, root = './data', 
                                    train = False, 
                                    transform = transforms.ToTensor(), 
                                    )
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]: 
        train_all_fine_map = train_dataset.targets
        train_sortidx = np.argsort(train_all_fine_map)
        train_sorted_fine_map = np.array(train_all_fine_map)[train_sortidx]
        # a dictionary that maps coarse id to a list of fine id
        target_fine_dict = {i:[] for i in range(len(train_dataset.coarse_names))}
        idx_train_source = [] # index of image (based on original Pytorch dataset) that sends to source
        idx_train_target = []
        f2c = dict(zip(range(len(train_dataset.fine_names)),train_dataset.coarse_map))
        for idx in range(len(train_sortidx)): # loop thru all argsort fine
            coarse_id = f2c[train_sorted_fine_map[idx]]
            target_fine_dict[coarse_id].append(train_sorted_fine_map[idx])
            if len(set(target_fine_dict[coarse_id])) <= 2: 
                # 2/5 to few shot second stage
                idx_train_target.append(train_sortidx[idx])
            else:
                # if we have seen the third type of fine class
                # since sorted, we have checked all images of the first
                # two types. For the rest of images,
                # send to source
                idx_train_source.append(train_sortidx[idx])
        
        for key in target_fine_dict:
            target = target_fine_dict[key] # fine label id for [coarse]
            d = {x: True for x in target}
            target_fine_dict[key] = list(d.keys())[:2] # all UNIQUE fine classes sent to target for [coarse]

        target_fine_cls = [] # all 4 fine classes sent to target
        for key in target_fine_dict:
            target_fine_cls.extend(target_fine_dict[key])

        test_all_fine_map = test_dataset.targets
        idx_test_source = []
        idx_test_target = []
        for idx in range(len(test_all_fine_map)):
            fine_id = test_all_fine_map[idx]
            coarse_id = f2c[fine_id]
            if fine_id in target_fine_dict[coarse_id]:
                idx_test_target.append(idx)
            else:
                idx_test_source.append(idx)


        source_fine_cls = list(set(range(len(train_dataset.fine_names))) - set(target_fine_cls))
        source_train = HierarchyMNISTSubset(case, idx_train_source, source_fine_cls, root = './data',
                                            train = True, transform = transforms.ToTensor())
        source_test = HierarchyMNISTSubset(case, idx_test_source, source_fine_cls, root = './data', 
                                           train = False, transform = transforms.ToTensor())
        target_train = HierarchyMNISTSubset(case, idx_train_target, target_fine_cls, root = './data',
                                            train = True, transform = transforms.ToTensor())
        target_test = HierarchyMNISTSubset(case, idx_test_target, target_fine_cls, root = './data', 
                                            train = False, transform = transforms.ToTensor())

        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    def make_inpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]:
        
        # for each fine class, there are n images
        # move n/2 to source and n/2 to target
        # classes for each split doesn't change
        
        fine_train_images = len(train_dataset) // len(train_dataset.fine_names)
        source_train, target_train = get_k_shot(fine_train_images//2, train_dataset, 'fine')
        
        fine_test_images = len(test_dataset) // len(test_dataset.fine_names)
        source_test, _ = get_k_shot(fine_test_images, test_dataset, 'fine') 
        target_test = source_test # don't touch test set, only convert to Subset class
        
        if task == 'ss':
            train_dataset, test_dataset = source_train, source_test
        elif task == 'st':
            train_dataset, test_dataset = source_train, target_test
        elif task == 'ts':
            train_dataset, test_dataset = target_train, source_test
        elif task == 'tt':
            train_dataset, test_dataset = target_train, target_test
        return train_dataset, test_dataset

    init_train_dataset, init_test_dataset = make_full_dataset()
    if task == 'sub_split_pretrain':
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'ss')
    elif task == 'sub_split_downstream': # where we use one shot setting for evaluation
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'tt')
    elif task == 'sub_split_zero_shot':
        train_dataset, test_dataset = make_subpopsplit_dataset(init_train_dataset, init_test_dataset, 'st')
    elif task == 'in_split_pretrain':
        train_dataset, test_dataset = make_inpopsplit_dataset(init_train_dataset, init_test_dataset, 'ss')
    elif task == 'in_split_downstream':
        train_dataset, test_dataset = make_inpopsplit_dataset(init_train_dataset, init_test_dataset, 'tt')
    elif task == 'in_split_zero_shot':
        train_dataset, test_dataset = make_inpopsplit_dataset(init_train_dataset, init_test_dataset, 'st')
    elif task == 'full':
        train_dataset, test_dataset = init_train_dataset, init_test_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
