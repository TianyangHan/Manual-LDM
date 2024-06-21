import torch
import torch.nn as nn
import numpy as np
import torchvision
from collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class vgg16(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False) :
        super(vgg16, self).__init__()

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices=5

        # add vgg pretrained features i into nn sequential with name 'i'
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
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,x):
        # output of vgg features
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class LPIPS(nn.Module):
    def __init__(self, net='vgg', version='0.1', use_dropout=True):
        super(LPIPS, self).__init__()
        self.version=version
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.scaling_layer = ScalingLayer()
        self.channels = [64,128,256,512,512]
        self.length = len(self.channels)
        self.L = len(self.channels)
        # Add 1x1 convolutional Layers
        self.lin0 = NetLinLayer(self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.channels[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)


        # Load the weights of trained LPIPS model
        import inspect
        import os
        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
        print('Loading model from: %s' % model_path)
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        ########################
        
        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        ########################

    def forward(self, x0,x1, normalize=False):
        # Scale the inputs to -1 to +1 range if needed
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            x0 = 2 * x0 - 1
            x1 = 2 * x1 - 1
        ########################
        # Normalize the inputs according to imagenet normalization
        x0_input, x1_input = self.scaling_layer(x0), self.scaling_layer(x1)

        # vgg outputs
        y0, y1 = self.net.forward(x0_input), self.net.forward(x1_input)
        feature0, feature1, diff = {},{},{}

        for k in range(self.length):
            feature0[k], feature1[k] =  nn.functional.normalize(y0[k], dim=1), nn.functional.normalize(y1[k])
            diff[k] = (y0[k] - y1[k]) ** 2


        res = [ spatial_average(self.lins[k](diff[k]), keepdim=True)  for k in range(self.L) ]
        return sum(res)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()

        # use register buffer to make sure they would not be updated during bp, and use None to fit any shape of the input x.
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
        

    def forward(self, inp):
        return (inp - self.shift)/self.scale


def spatial_average(input, keepdim=True):
    # B C H W -> B C 1 1
    return input.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    
    def __init__(self, in_channels, out_channels=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out