from torch import Tensor
import torch.nn as nn
from typing import *


class CNN(nn.Module): 
    '''
    CNN model.
    '''
    def __init__(self, num_classes : List[int]):
        super(CNN, self).__init__()
        self.base = nn.Sequential(nn.Conv2d(1,6,5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(6,16,5),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.Flatten(1),
                                  nn.Linear(256,120),
                                  nn.ReLU(),
                                  nn.Linear(120,50),
                                  nn.ReLU())
        self.out_features = 50
        self.head = len(num_classes)
        if self.head == 1:
            self.fc = nn.Linear(self.out_features, num_classes[0]) 
        elif self.head == 2:
            num_coarse_classes = num_classes[0]
            num_fine_classes = num_classes[1]
            assert (num_coarse_classes <= num_fine_classes), 'invalid hierarchy'
            self.fc1 = nn.Linear(self.out_features, num_coarse_classes) 
            self.fc2 = nn.Linear(self.out_features, num_fine_classes)
        
    
    def forward(self, x : Tensor) -> Tuple[Tensor, ...]:
        x = self.base(x)
        if self.head == 1:
            return x, self.fc(x)
        elif self.head == 2:
            return x, self.fc1(x), self.fc2(x)