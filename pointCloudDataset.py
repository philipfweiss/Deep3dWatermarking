import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

dataset_folder = "/home/jlipman500/ShapeNetCore.slice"
dataset_folder = "dataset/"
np_file_format = "model_normalized.npy"

class PointCloudDataset(Dataset):

    """
    Loads the dataset
    """
    def __init__(self):
        '''
        Open python object that points to dataset directory for iteration
        save as instance variable
        '''
        self.objects = os.listdir(dataset_folder)
        self.use_cuda = torch.cuda.is_available()
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
        print('foo')
        object = self.objects.__getitem__(idx)
        obj_file_name = os.path.join(dataset_folder, object)#, "models", np_file_format)
        numpy_file = np.load(obj_file_name)
        if self.use_cuda:
            tensor = torch.from_numpy(numpy_file).float().cuda()
        else:
            tensor = torch.from_numpy(numpy_file).float()
        tensor = tensor[np.newaxis, ...]
        return tensor
