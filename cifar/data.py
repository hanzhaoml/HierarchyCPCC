from typing import *

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from hierarchy.data import Hierarchy, get_k_shot

class HierarchyCIFAR(CIFAR100, Hierarchy):
    def __init__(self, *args, **kw):
        super(HierarchyCIFAR, self).__init__(*args, **kw)
        self.fine_map = np.array(range(100))
        # coarse_map = mapping fine id to coarse id
        # fine coarse id labeled alphabetically
        self.coarse_map = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.fine_names = ["apple","aquarium_fish","baby","bear","beaver",
                    "bed","bee","beetle","bicycle","bottle",
                    "bowl","boy","bridge","bus","butterfly",
                    "camel","can","castle","caterpillar","cattle",
                    "chair","chimpanzee","clock","cloud","cockroach",
                    "couch","crab","crocodile","cup","dinosaur",
                    "dolphin","elephant","flatfish","forest","fox",
                    "girl","hamster","house","kangaroo","keyboard",
                    "lamp","lawn_mower","leopard","lion","lizard",
                    "lobster","man","maple_tree","motorcycle","mountain",
                    "mouse","mushroom","oak_tree","orange","orchid",
                    "otter","palm_tree","pear","pickup_truck","pine_tree",
                    "plain","plate","poppy","porcupine","possum",
                    "rabbit","raccoon","ray","road","rocket",
                    "rose","sea","seal","shark","shrew",
                    "skunk","skyscraper","snail","snake","spider",
                    "squirrel","streetcar","sunflower","sweet_pepper","table",
                    "tank","telephone","television","tiger","tractor",
                    "train","trout","tulip","turtle","wardrobe",
                    "whale","willow_tree","wolf","woman","worm"]
        self.coarse_names = ["aquatic_mammals","fish",
                    "flowers","food_containers","fruit_and_vegetabes",
                    "household_electrical_devices",
                    "household_furniture","insects","large_carnivores",
                    "large_man-made_outdoor_things",
                    "large_natural_outdoor_scenes",
                    "large_omnivores_and_herbivores","medium_mammals",
                    "non-insect_invertebrates","people",
                    "reptiles","small_mammals","trees","vehicles_1","vehicles_2"]
        self.img_size = 32
        self.channel = 3
        self.mid_map = np.zeros((len(self.fine_map),)) # fine to mid map
        for i in range(len(self.coarse_names)):
            i_coarse_indices = (self.coarse_map == i).nonzero()[0]
            two_classes = i_coarse_indices[:2] # first 2 => mid_n+1
            self.mid_map[two_classes] = i + 0.5
            three_classes = i_coarse_indices[2:] # last 3 => mid_n
            self.mid_map[three_classes] = i
        self.mid_map *= 2
        self.mid_names = sorted([x + "_a" for x in self.coarse_names] + [x + "_b" for x in self.coarse_names])
        self.mid_map = self.mid_map.astype(int)

        self.coarsest_map = np.zeros((len(self.fine_map),)) # fine to coarsest map
        self.coarsest_names = []
        for i in range(0,len(self.coarse_names),2):
            i_group_indices = list((self.coarse_map == i).nonzero()[0]) + list(((self.coarse_map == (i+1)).nonzero()[0]))
            self.coarsest_map[i_group_indices] = i // 2
            self.coarsest_names.append((self.coarse_names[i] + "_" + self.coarse_names[i+1]))
        self.coarsest_map = self.coarsest_map.astype(int)
    
    def __getitem__(self, index: int):
        img, target_fine = self.data[index], int(self.targets[index])

        target_mid = int(self.mid_map[target_fine])
        target_coarse = int(self.coarse_map[target_fine])
        target_coarsest = int(self.coarsest_map[target_fine])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_fine = self.target_transform(target_fine)
        return img, target_coarsest, target_coarse, target_mid, target_fine


class HierarchyCIFARSubset(HierarchyCIFAR):
    '''
        Reset label index from 0 for source and target datasest.
        Args:
            indices: list of subset indices on HierarchyCIFAR
            fine_classes: which fine classes are included in this subset
        '''
    def __init__(self, indices : List[int], fine_classes : List[int], *args, **kw):
        super(HierarchyCIFARSubset, self).__init__(*args, **kw)
        self.data = self.data[indices]
        
        old_targets = list(np.array(self.targets)[indices]) # old fine targets, sliced
        fine_classes = np.array(sorted(fine_classes)) # fine class id in HierarchyCIFAR index, range = 0-19
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

        old_coarsest_map = np.array([self.coarsest_map[i] for i in fine_classes])
        coarsest_classes = np.unique(old_coarsest_map)
        self.coarsest_names = [self.coarsest_names[cls] for cls in coarsest_classes]
        self.coarsest_map = np.array([list(coarsest_classes).index(i) for i in old_coarsest_map])

    def __len__(self):
        return len(self.targets)

def make_dataloader(num_workers : int, batch_size : int, task : str) -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader
        task : if 'split_pretrain', dataset has 60 classes, if 'split_downstream',
        dataset has 40 classes, if 'full', dataset has 100 classes.
    '''
    def make_full_dataset() -> Tuple[Dataset, Dataset]:
        '''
        Create full size augmented CIFAR100 dataset that contains four layers in
        hierarchy labels. Return a tuple of train, test dataset.
        '''
        train_dataset = HierarchyCIFAR(root = './data',
                                    train = True,                         
                                    transform = transforms.Compose(
                                                    [transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], 
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])        
                                    )
        # not augment test set except of normalization
        test_dataset = HierarchyCIFAR(root = './data', 
                                    train = False, 
                                    transform = transforms.Compose(
                                                    [transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], 
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])
                                    )
        return train_dataset, test_dataset

    def make_subpopsplit_dataset(train_dataset : Hierarchy, test_dataset : Hierarchy, task : str) -> Tuple[DataLoader, DataLoader]: 
        '''
        Given full size CIFAR100 datasets, split fine classes into 40/60. Specifically,
        For each coarse class, its 5 fine children is divided into 2 (target) and 3 (source)
        in an increasing alphabetical order. Finally, we have 60 fine classes in source,
        40 fine classes in target. The same procedure is applied on full size train and 
        test set. 
        Depending on the task, we use different source/target train/test combination. For
        pretrain, we use 'ss'; for evaluation of representation on one-shot transfer, we 
        use 'tt'.
        Args:
            train_dataset : augmented CIFAR100 train dataset with four label levels
            test_dataset : normalized CIFAR100 test dataset with four label levels
            task : 'ss' = 'source train source val', 'st' = 'source train target val',
            'ts' = 'target train source val', 'tt' = 'target train target val'
        '''
        train_all_fine_map = train_dataset.targets
        train_sortidx = np.argsort(train_all_fine_map)
        train_sorted_fine_map = np.array(train_all_fine_map)[train_sortidx]
        # a dictionary that maps coarse id to a list of fine id
        target_fine_dict = {i:[] for i in range(len(train_dataset.coarse_names))}
        idx_train_source = [] # index of image (based on original Pytorch CIFAR dataset) that sends to source
        idx_train_target = []
        f2c = dict(zip(range(len(train_dataset.fine_names)),train_dataset.coarse_map))
        for idx in range(len(train_sortidx)): # loop thru all argsort fine
            coarse_id = f2c[train_sorted_fine_map[idx]]
            target_fine_dict[coarse_id].append(train_sorted_fine_map[idx])
            if len(set(target_fine_dict[coarse_id])) <= 2: 
                # 40% = 2/5 to few shot second stage
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

        target_fine_cls = [] # all 40 fine classes sent to target
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
        source_train = HierarchyCIFARSubset(idx_train_source, source_fine_cls, root = './data',
                                train = True,                         
                                transform = transforms.Compose(
                                                    [
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation(15),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean = [0.5071, 0.4867, 0.4408], 
                                                            std = [0.2675, 0.2565, 0.2761],
                                                        )
                                                    ]))
        source_test = HierarchyCIFARSubset(idx_test_source, source_fine_cls, root = './data', 
                                    train = False, 
                                    transform = transforms.Compose(
                                                    [
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean = [0.5071, 0.4867, 0.4408], 
                                                            std = [0.2675, 0.2565, 0.2761],
                                                        )
                                                    ]))
        target_train = HierarchyCIFARSubset(idx_train_target, target_fine_cls, root = './data',
                                train = True,                         
                                transform = transforms.Compose(
                                                    [
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation(15),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean = [0.5071, 0.4867, 0.4408], 
                                                            std = [0.2675, 0.2565, 0.2761],
                                                        )
                                                    ]))
        target_test = HierarchyCIFARSubset(idx_test_target, target_fine_cls, root = './data', 
                                        train = False, 
                                        transform = transforms.Compose(
                                                        [
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(
                                                                mean = [0.5071, 0.4867, 0.4408], 
                                                                std = [0.2675, 0.2565, 0.2761],
                                                            )
                                                        ]))
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

    def make_outlier_dataset() -> Tuple[Dataset, Dataset]:
        '''
        Create full size augmented CIFAR10 dataset. 
        Augmentation the same as CIFAR100.
        Return a tuple of train, test dataset.
        '''
        train_dataset = CIFAR10(root = './data',
                                    train = True,                         
                                    transform = transforms.Compose(
                                                    [transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], # use CIFAR100's mean/std
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])        
                                    )
        test_dataset = CIFAR10(root = './data', 
                                    train = False, 
                                    transform = transforms.Compose(
                                                    [transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean = [0.5071, 0.4867, 0.4408], 
                                                        std = [0.2675, 0.2565, 0.2761],
                                                    )
                                                    ])
                                    )
        fine_names = ['airplane','automobile','bird','cat','deer',
                      'dog','frog','horse','ship','truck']
        train_dataset.fine_names = fine_names
        test_dataset.fine_names = fine_names
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
    elif task == 'outlier':
        train_dataset, test_dataset = make_outlier_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
