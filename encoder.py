from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        self.bn1 = nn.BatchNorm3d(1)
        self.bn2 = nn.BatchNorm3d(1)
        self.bn3 = nn.BatchNorm3d(10)
        self.bn4 = nn.BatchNorm3d(1)
        self.bn5 = nn.BatchNorm3d(10)
        self.bn6 = nn.BatchNorm3d(1)

        self.conv1 = nn.Conv3d(1, 1, 3, 1, 1)
        self.conv2 = nn.Conv3d(1, 1, 3, 1, 1)
        self.conv3 = nn.Conv3d(1+k, 10, 3, 1, 1)
        self.conv4 = nn.Conv3d(10, 1, 3, 1, 1)
        self.conv5 = nn.Conv3d(1, 10, 3, 1, 1)
        self.conv6 = nn.Conv3d(10, 1, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        gaussian_kernel = torch.ones(3,3,3)/27
        gaussian_kernel = gaussian_kernel[None,None,...]
        self.blend = nn.Conv3d(1, 1, 3, 1,1,groups = 1, bias = False)
        self.blend.weight.data = gaussian_kernel
        self.blend.weight.requires_grad = False

    def forward(self, x, message, mask):
        ## Begin by encoding x with 2 conv-bn-relu blocks.
        intermediate = self.leaky_relu(self.bn1(self.conv1(x)))
        intermediate = self.leaky_relu(self.bn2(self.conv2(intermediate)))
        ## Concat x and message
        mask = self.blend(self.blend(mask))
        message = message*mask
        concated = torch.cat((intermediate, message), 1)

        ## more conv layers
        encoded = self.leaky_relu(self.bn3(self.conv3(concated)))
        encoded = self.leaky_relu(self.bn4(self.conv4(encoded)))

        # encoded *= mask
        encoded = encoded / torch.sum(encoded)

        skip_connection = encoded + x
        final = self.leaky_relu(self.bn5(self.conv5(skip_connection)))
        final = self.leaky_relu(self.bn6(self.conv6(final)))

        # final *= mask
        # final *= 1/torch.sum(final)

        return final
