import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

dataset_folder = "/home/jlipman500/ShapeNetCore.slice"
np_file_format = "model_normalized.npy"

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
        return len([name for name in os.listdir(dataset_folder)])

    """
    Grabs the idx-th item from the dataset.
    """
    def __getitem__(self, idx):
        objects = os.listdir(dataset_folder)
        object = objects.__getitem__(idx)
        obj_file_name = os.path.join(dataset_folder, object, "model", np_file_format)
        numpy_file = np.load(obj_file_name)
        tensor = torch.from_numpy(numpy_file).float().cuda()
        return tensor

