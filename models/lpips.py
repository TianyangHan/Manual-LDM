import torch
import torch.nn as nn
import numpy as np
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class vgg16(nn.module):
    def __init__(self, pretrained=True, requires_grad=False) :
        super(vgg16, self).__init__()

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices=5

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4,9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9,16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16,23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23,30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # freeze vgg16 ori weights
        if not requires_grad:
            for param in self.para