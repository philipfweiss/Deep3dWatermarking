from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(13, 10, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 10, 3, 1, 1)

    def forward(self, x, message):

        ## Begin by encoding x with 2 conv-bn-relu blocks.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        encoding = x.clone()
        ## Concat x and message
        x = torch.cat((x, message), 1)

        ## Six more conv layers
        x = F.relu(self.bn3(self.conv3(x)))

        return x, encoding
