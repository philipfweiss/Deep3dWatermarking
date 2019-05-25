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
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(10)
        self.bn4 = nn.BatchNorm2d(3)
        self.bn5 = nn.BatchNorm2d(10)
        self.bn6 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(13, 10, 3, 1, 1)
        self.conv4 = nn.Conv2d(10, 3, 3, 1, 1)
        self.conv5 = nn.Conv2d(3, 10, 3, 1, 1)
        self.conv6 = nn.Conv2d(10, 3, 3, 1, 1)

    def forward(self, x, message):

        ## Begin by encoding x with 2 conv-bn-relu blocks.
        intermediate = F.relu(self.bn1(self.conv1(x)))
        intermediate = F.relu(self.bn2(self.conv2(intermediate)))

        #intermediate = x.clone()
        ## Concat x and message
        concated = torch.cat((intermediate, message), 1)

        ## more conv layers
        encoded = F.relu(self.bn3(self.conv3(concated)))
        encoded = F.relu(self.bn4(self.conv4(encoded)))

        skip_connection = encoded + x
        final = F.relu(self.bn5(self.conv5(skip_connection)))
        final = F.relu(self.bn6(self.conv6(final)))

        return final
