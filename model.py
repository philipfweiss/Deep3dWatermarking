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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 400)
        self.fc2 = nn.Linear(400, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
