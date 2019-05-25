from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.conv5 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv6 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv7 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv8 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv9 = nn.Conv2d(10, 10, 3, 1, 1)
        self.conv10 = nn.Conv2d(10, 10, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(10)
        self.bn6 = nn.BatchNorm2d(10)
        self.bn7 = nn.BatchNorm2d(10)
        self.bn8 = nn.BatchNorm2d(10)
        self.bn9 = nn.BatchNorm2d(10)
        self.bn10 = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(640, 10)

    def forward(self, x):

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))

        x = self.pool(x)


        ## Flatten and affine
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)

        return x, encoding
