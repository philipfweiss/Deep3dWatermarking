from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

"""
Our model has the following structure:
    - Encoder (E) encodes an input model with a message K.
    - Projector (P) Creates d 2D projections from the encoded mesh.
    - Decoder (D) reconstructs message K from projections.
    - Adversary (A) attempts to discern encoded from non-incoded projections.
"""

class Net(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
