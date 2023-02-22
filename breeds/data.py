from torch.utils.data import DataLoader, Subset

from robustness.tools.folder import ImageFolder
from robustness.datasets import CustomImageNet
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

import os
from typing import *
import numpy as np
import json

from hierarchy.data import Hierarchy


class HierarchyBREEDS(ImageFolder, Hierarchy):
    def __init__(self, info_dir, coarse2coarsest_wnid, fine2mid_wnid, coarse_id2name, *args, **kw):
        super(HierarchyBREEDS, self).__init__(*args, **kw)
        self.fine_names = sorted(list(self.class_to_idx.keys()))
        self.mid_names = sorted(list(set(fine2mid_wnid.values()))) 
        self.coarse_names = list(set(self.targets))
        self.coarsest_names = sorted(list(set(coarse2coarsest_wnid.values())))
        
        self.coarse_id2name = coarse_id2name # ret[-1], coarse target id to name
        
        with open(os.path.join(info_dir, "node_names.txt")) as f:
            wnid2name = [l.strip().split('\t') for l in f.readlines()]
        
        name2wnid = dict() # name to wnid
        for pairs in wnid2name:
            wnid = pairs[0]
            name = pairs[1]
            if not name in name2wnid:
                name2wnid[name] = [wnid]
            else:
                name2wnid[name].append(wnid)
        self.name2wnid = name2wnid
        self.mid_map = [0] * len(self.fine_names)
        for fine_idx in range(len(self.fine_names)):
            target_fine_wnid = self.fine_names[fine_idx]
            target_mid_wnid = fine2mid_wnid[target_fine_wnid]
            self.mid_map[fine_idx] = self.mid_names.index(target_mid_wnid)
        self.mid_map = np.array(self.mid_map)
        
        self.coarse2coarsest_idx = [0] * len(self.coarse_names)
        for coarse_idx in range(len(set(self.class_to_idx.values()))):
            target_coarse_name = self.coarse_id2name[coarse_idx]
            target_coarse_wnids = self.name2wnid[target_coarse_name]
            if len(target_coarse_wnids) == 1:
                target_coarse_wnid = target_coarse_wnids[0]
            else:
                target_coarse_wnid_candid = set(coarse2coarsest_wnid.keys())
                target_coarse_wnid = list(set(target_coarse_wnids).intersection(target_coarse_wnid_candid))[0]
            target_coarsest_wnid = coarse2coarsest_wnid[target_coarse_wnid]
            self.coarse2coarsest_idx[coarse_idx] = self.coarsest_names.index(target_coarsest_wnid)
        
        coarse_map = [0] * len(self.fine_names)
        coarsest_map = [0] * len(self.fine_names)
        all_fine_targets = [self.fine_names.index(self.samples[i][0].split("/")[-2]) for i in range(len(self.samples))]
        all_coarse_targets = self.targets
        all_coarsest_targets = [self.coarse2coarsest_idx[i] for i in self.targets]
        for i in range(len(coarse_map)):
            fine_idx = i
            fine_key = all_fine_targets.index(fine_idx)
            coarse_id = all_coarse_targets[fine_key]
            coarsest_id = all_coarsest_targets[fine_key]
            coarse_map[i] = coarse_id
            coarsest_map[i] = coarsest_id
        
        self.coarse_map = np.array(coarse_map)
        self.coarsest_map = np.array(coarsest_map)
        self.targets = all_fine_targets
    
    def __getitem__(self, index: int):
        # let get item be consistent with other datasets
        path, _ = self.samples[index]
        target_fine = int(self.targets[index])

        target_mid = int(self.mid_map[target_fine])
        target_coarse = int(self.coarse_map[target_fine])
        target_coarsest = int(self.coarsest_map[target_fine])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target_coarsest, target_coarse, target_mid, target_fine

def make_dataloader(num_workers : int, batch_size : int, task : str, breeds_setting: str, 
                    info_dir : str = './breeds/imagenet_class_hierarchy/modified', 
                    data_dir : str = '/data/common/ILSVRC2012') -> Tuple[DataLoader, DataLoader]:
    '''
    Args:
        breeds_setting : living17/entity13/entity30/nonliving26
    '''
    def make_edge_mapping(info_dir, superclasses : list):
        '''
            Backtrack one edge from node to its parent
        '''
        with open(os.path.join(info_dir, "class_hierarchy.txt")) as f:
            edges = [l.strip().split() for l in f.readlines()]
        mapping = dict()
        # each node except for the root has one parent node
        for pairs in edges:
            start, end = pairs[0], pairs[1]
            if end in mapping:
                import pdb; pdb.set_trace()
            else:
                if end in superclasses:
                    mapping[end] = start
        return mapping

    def make_loaders(info_dir, workers, batch_size, supsup_mapping, mid_mapping, name_mapping, transforms, data_path, data_aug=True,
                custom_class=None, dataset="", label_mapping=None, subset=None,
                subset_type='rand', subset_start=0, val_batch_size=None,
                only_val=False, shuffle_train=True, shuffle_val=True, seed=1,
                custom_class_args=None):
        '''
            make PreHierarchyBREEDS train/test loaders.
        '''
        print(f"==> Preparing dataset {dataset}..")
        transform_train, transform_test = transforms
        if not data_aug:
            transform_train = transform_test

        if not val_batch_size:
            val_batch_size = batch_size

        if not custom_class:
            train_path = os.path.join(data_path, 'train')
            test_path = os.path.join(data_path, 'val')
            if not os.path.exists(test_path):
                test_path = os.path.join(data_path, 'test')

            if not os.path.exists(test_path):
                raise ValueError("Test data must be stored in dataset/test or {0}".format(test_path))
            
            if not only_val:
                train_set = HierarchyBREEDS(info_dir, supsup_mapping, mid_mapping, name_mapping, root=train_path, transform=transform_train,
                                            label_mapping=label_mapping)
            test_set = HierarchyBREEDS(info_dir, supsup_mapping, mid_mapping, name_mapping, root=test_path, transform=transform_test,
                                        label_mapping=label_mapping)
           
        else:
            if custom_class_args is None: custom_class_args = {}
            if not only_val:
                train_set = custom_class(root=data_path, train=True, download=True, 
                                    transform=transform_train, **custom_class_args)
            test_set = custom_class(root=data_path, train=False, download=True, 
                                    transform=transform_test, **custom_class_args)

        if not only_val:
            attrs = ["samples", "train_data", "data"]
            vals = {attr: hasattr(train_set, attr) for attr in attrs}
            assert any(vals.values()), f"dataset must expose one of {attrs}"
            train_sample_count = len(getattr(train_set,[k for k in vals if vals[k]][0]))

        if (not only_val) and (subset is not None) and (subset <= train_sample_count):
            assert not only_val
            if subset_type == 'rand':
                rng = np.random.RandomState(seed)
                subset = rng.choice(list(range(train_sample_count)), size=subset+subset_start, replace=False)
                subset = subset[subset_start:]
            elif subset_type == 'first':
                subset = np.arange(subset_start, subset_start + subset)
            else:
                subset = np.arange(train_sample_count - subset, train_sample_count)

            train_set = Subset(train_set, subset)

        if not only_val:
            train_loader = DataLoader(train_set, batch_size=batch_size, 
                shuffle=shuffle_train, num_workers=workers, pin_memory=True)

        test_loader = DataLoader(test_set, batch_size=val_batch_size, 
                shuffle=shuffle_val, num_workers=workers, pin_memory=True)

        if only_val:
            return None, test_loader

        return train_loader, test_loader

    if breeds_setting == 'living17':
        ret = make_living17(info_dir, split="rand")
    elif breeds_setting == 'entity13':
        ret = make_entity13(info_dir, split="rand")
    elif breeds_setting == 'entity30':
        ret = make_entity30(info_dir, split="rand")
    elif breeds_setting == 'nonliving26':
        ret = make_nonliving26(info_dir, split="rand")
    superclasses, subclass_split, label_map = ret 
    supsup_mapping = make_edge_mapping(info_dir, superclasses)
    train_subclasses, test_subclasses = subclass_split

    dataset_source = CustomImageNet(data_dir, train_subclasses)
    index2subname = json.load(open("./breeds/imagenet_class_hierarchy/imagenet_index_map.json"))
    index2subname = {int(key):index2subname[key] for key in index2subname}
    all_sub_wnid = [row.split("\n")[0].split(" ")[0] for row in open("./breeds/imagenet_class_hierarchy/imagenet_wnid.txt")]
    train_flattened_subclasses = [idx for split in train_subclasses for idx in split] # label index
    train_subclasses_names = [index2subname[sub] for sub in train_flattened_subclasses]
    
    with open(os.path.join(info_dir, "node_names.txt")) as f:
        wnid2name = [l.strip().split('\t') for l in f.readlines()]
        name2wnid = dict() # name to wnid
        for pairs in wnid2name:
            wnid = pairs[0]
            name = pairs[1]
            if wnid in all_sub_wnid:
                name2wnid[name] = wnid 
    
    train_subclasses_wnids = [name2wnid[name] for name in train_subclasses_names]
    source_mid_mapping = make_edge_mapping(info_dir, train_subclasses_wnids)
    source_transforms = (dataset_source.transform_train, dataset_source.transform_test)
    loaders_source = make_loaders(info_dir, num_workers, batch_size, supsup_mapping, source_mid_mapping, label_map, transforms=source_transforms,
                                    data_path=dataset_source.data_path,
                                    dataset=dataset_source.ds_name,
                                    label_mapping=dataset_source.label_mapping,
                                    custom_class=dataset_source.custom_class,
                                    custom_class_args=dataset_source.custom_class_args)
    train_loader_source, val_loader_source = loaders_source # val_loader_source for s->s exp

    dataset_target = CustomImageNet(data_dir, test_subclasses)
    source_mid_mapping = make_edge_mapping(info_dir, train_subclasses_wnids)
    test_flattened_subclasses = [idx for split in test_subclasses for idx in split] # label index
    test_subclasses_names = [index2subname[sub] for sub in test_flattened_subclasses]
    test_subclasses_wnids = [name2wnid[name] for name in test_subclasses_names]
    target_mid_mapping = make_edge_mapping(info_dir, test_subclasses_wnids)
    target_transforms = (dataset_target.transform_train, dataset_target.transform_test)
    loaders_target = make_loaders(info_dir, num_workers, batch_size, supsup_mapping, target_mid_mapping, label_map, transforms=target_transforms,
                                    data_path=dataset_target.data_path,
                                    dataset=dataset_target.ds_name,
                                    label_mapping=dataset_target.label_mapping,
                                    custom_class=dataset_target.custom_class,
                                    custom_class_args=dataset_target.custom_class_args)
    train_loader_target, val_loader_target = loaders_target
    
    if task == 'sub_split_pretrain':
        train_loader, test_loader = train_loader_source, val_loader_source
    elif task == 'sub_split_downstream': # where we use one shot setting for evaluation
        train_loader, test_loader = train_loader_target, val_loader_target
    elif task == 'sub_split_zero_shot':
        train_loader, test_loader = train_loader_source, val_loader_target
    elif task == 'full': # use source/source as full data
        train_loader, test_loader = train_loader_source, val_loader_source
    return train_loader, test_loader