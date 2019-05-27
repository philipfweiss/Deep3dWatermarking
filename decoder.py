from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, k):
        super(Decoder, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.conv2 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv3 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv4 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv5 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv6 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv7 = nn.Conv2d(10, 10, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(10)
        self.bn4 = nn.BatchNorm2d(10)
        self.bn5 = nn.BatchNorm2d(10)
        self.bn6 = nn.BatchNorm2d(10)
        self.bn7 = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(640, k)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        ## Flatten and affine
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(x)
        # x = self.fc2(x)
        return x
