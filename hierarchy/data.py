from torch.utils.data import Subset
import numpy as np
from nltk.tree import Tree

import random
from typing import *

class Hierarchy():
    '''
        Generic hierarchy class with 4 levels of layers.
    '''
    def __init__(self, *args, **kw):
        super(Hierarchy, self).__init__(*args, **kw)
        # === fine to layer map ===
        self.fine_map = np.empty(0)
        self.mid_map = np.empty(0)
        self.coarse_map = np.empty(0)
        self.coarsest_map = np.empty(0)
        # === names of each layer ===
        self.fine_names : List[str] = []
        self.mid_names : List[str] = []
        self.coarse_names : List[str] = []
        self.coarsest_names : List[str] = []

    def fine2mid(self):
        return dict(zip(range(len(self.fine_names)), self.mid_map)) 
    
    def fine2coarse(self):
        return dict(zip(range(len(self.fine_names)), self.coarse_map))
    
    def fine2coarsest(self):
        return dict(zip(range(len(self.fine_names)), self.coarsest_map))
    
    def build_nltk_tree(self):
        # root, coarse, fine
        coarse_level = [None for _ in range(len(self.coarse_names))]
        for i in range(len(coarse_level)):
            fine_level = [None for _ in range(len(self.fine_names)//len(self.coarse_names))]
            fine_idx = (self.coarse_map == i).nonzero()[0]
            for j in range(len(fine_level)):
                fine_name = self.fine_names[fine_idx[j]]
                fine_level[j] = fine_name
            coarse_level[i] = Tree(self.coarse_names[i],fine_level)
        return Tree('root',coarse_level)

    def make_distances(self):
        # height of our tree is always 3 = [coarse + fine + root]
        height = 3
        distances = dict()
        for i in range(len(self.fine_names)):
            for j in range(len((self.fine_names))):
                if i == j:
                    distances[i,j] = 0
                else:
                    if self.coarse_map[i] == self.coarse_map[j]:
                        lca_h = 1
                    else:
                        lca_h = 2
                    distances[i,j] = lca_h/height
        return distances

def get_k_shot(k : int, Hierarchy : Hierarchy, layer : str, seed = None) -> Tuple[Subset, Subset]: 
    '''
    Get k images for each of the layer class. Return a subset of CIFAR dataset.
    Args:
        k : number of images selected for each class in layer 
        HierarchyCifar : CIFAR100 with 4 hierarchical labels
        layer : apply k shot to fine/mid/coarse/coarsest
    '''
    all_fine_targets = Hierarchy.targets
    # we want to convert fine labels to layer labels
    if layer == 'fine':
        converted_targets = all_fine_targets
        layer_size = len(Hierarchy.fine_names)
    else:
        if layer == 'mid':
            converted_dict = Hierarchy.fine2mid()
            layer_size = len(Hierarchy.mid_names)
        elif layer == 'coarse':
            converted_dict = Hierarchy.fine2coarse()
            layer_size = len(Hierarchy.coarse_names)
        elif layer == 'coarsest':
            converted_dict = Hierarchy.fine2coarsest()
            layer_size = len(Hierarchy.coarsest_names)
        converted_targets = np.array([converted_dict[x] for x in all_fine_targets])
    all_layer_num_images = np.unique(converted_targets, return_counts=True)[1] # unique counts sorted
    # first k images, for each class in the layer
    # this is the index assuming converted target is sorted by layer class id
    if seed is None:
        k_img_idx = [sum(all_layer_num_images[:i])+j for i in range(layer_size) for j in range(k)] 
    else:
        k_img_idx = []
        for i in range(layer_size):
            lo = sum(all_layer_num_images[:i])
            hi = sum(all_layer_num_images[:(i+1)]) # exclusive
            random.seed(seed)
            indices = random.sample(list(range(lo,hi)), k)
            k_img_idx.extend(indices)
    rest_idx = list(set(range(len(all_fine_targets))) - set(k_img_idx))
    sortidx = np.argsort(converted_targets)
    return Subset(Hierarchy, sortidx[k_img_idx]), Subset(Hierarchy, sortidx[rest_idx])