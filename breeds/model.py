import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet18

from typing import *

class ResNet18(nn.Module): 
    
    def __init__(self, num_classes : List[int]):
        super(ResNet18, self).__init__()
        self.backbone = resnet18()  # for imagenet
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
        self.out_features = 512
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
        # representation vector, returned here to avoid extra gpu usage
        x = torch.squeeze(self.base(x)) 
        if self.head == 1:
            return x, self.fc(x)
        elif self.head == 2:
            return x, self.fc1(x), self.fc2(x)