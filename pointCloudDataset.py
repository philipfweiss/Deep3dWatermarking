import torch
from torch.utils.data import Dataset, DataLoader

## We need to define __len__ and a __getitem__
class PointCloudDataset(Dataset):

    """
    Loads the dataset
    """
    def __init__(self):
        pass

    """
    Returns the total number of examples in the dataset.
    """
    def __len__(self):
        pass

    """
    Grabs the idx-th item from the dataset.
    """
    def __getitem__(self, idx):
        pass
