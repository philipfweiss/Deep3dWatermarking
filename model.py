from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

"""
Our model has the following structure:
    - Encoder (E) encodes an input model with a message K.
    - Projector (P) Creates d 2D projections from the encoded mesh.
    - Decoder (D) reconstructs message K from projections.
    - Adversary (A) attempts to discern encoded from non-incoded projections.
"""

class MeshWatermarker(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(13, 10, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 10, 3, 1, 1)

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


        self.bn4 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(640, 10)
        # self.fc2 = nn.Linear(600, 10)

    def forward(self, x, message):

        ## Begin by encoding x with 2 conv-bn-relu blocks.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        encoding = x.clone()
        ## Concat x and message
        x = torch.cat((x, message), 1)

        ## Six more conv layers
        x = F.relu(self.bn3(self.conv3(x)))
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
