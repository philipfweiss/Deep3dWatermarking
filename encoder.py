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
        '''
        # Set these to whatever you want for your gaussian filter
        kernel_size = 3
        sigma = 3
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        z_grid = x_grid
        print(torch.stack([x_grid, y_grid], dim=-1))
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        print(xyz_grid)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
        print(gaussian_kernel.shape)


        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        print(gaussian_kernel.shape)
        '''
        gaussian_kernel = torch.ones(3,3,3)/27
        gaussian_kernel = gaussian_kernel[None,None,...]
        self.blend = torch.nn.Conv3d(1, 1, 3, 1, 1)
        self.blend.weight.data = gaussian_kernel
        self.blend.weight.requires_grad = False

    def forward(self, x, message, mask):

        ## Begin by encoding x with 2 conv-bn-relu blocks.
        intermediate = self.leaky_relu(self.bn1(self.conv1(x)))
        intermediate = self.leaky_relu(self.bn2(self.conv2(intermediate)))
        print(self.conv1.weight.data.shape)
        #intermediate = x.clone()
        ## Concat x and message
        concated = torch.cat((intermediate, message), 1)

        ## more conv layers
        encoded = self.leaky_relu(self.bn3(self.conv3(concated)))
        encoded = self.leaky_relu(self.bn4(self.conv4(encoded)))

        skip_connection = encoded + x
        final = self.leaky_relu(self.bn5(self.conv5(skip_connection)))
        final = self.leaky_relu(self.bn6(self.conv6(final)))

        mask = self.blend(mask)
        print(self.blend.weight.data)
        final *= mask

        return final
