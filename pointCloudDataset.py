import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

np_file_format = "model_normalized.npy"

class PointCloudDataset(Dataset):

    """
    Loads the dataset
    """
    def __init__(self, localdev):
        '''
        Open python object that points to dataset directory for iteration
        save as instance variable
        '''
        if localdev:
            self.dataset_folder = "dataset/"
        else:
            self.dataset_folder = "/home/jlipman500/ShapeNetCore.slice"
        self.objects = os.listdir(self.dataset_folder)
        self.use_cuda = torch.cuda.is_available()
        self.local = localdev


    """
    Returns the total number of examples in the dataset.
    """
    def __len__(self):
        return 100#len([name for name in os.listdir(self.dataset_folder)])

    """
    Grabs the idx-th item from the dataset.
    """
    def __getitem__(self, idx):
        object = self.objects.__getitem__(idx)
        if self.local:
            obj_file_name = os.path.join(self.dataset_folder, object)#, "models", np_file_format)
        else:
            obj_file_name = os.path.join(self.dataset_folder, object, "models", np_file_format)
        numpy_file = np.load(obj_file_name)
        tensor = torch.from_numpy(numpy_file).float()
        tensor = tensor[np.newaxis, ...]
        return tensor
