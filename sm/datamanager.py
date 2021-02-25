import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


# MNIST scenarios
scenarios = {
    "1": {
        "inlier_classes": [1],
        "outlier_classes": [0],
        "prefix": "scenario-1"
    },
    "2": {
        "inlier_classes": [0, 1],
        "outlier_classes": [2, 3],
        "prefix": "scenario-2"
    },
    "3": {
        "inlier_classes": [0, 1, 2, 3],
        "outlier_classes": [4, 5, 6],
        "prefix": "scenario-3"
    },
    "4": {
        "inlier_classes": [0, 1, 2, 3, 4, 5, 6],
        "outlier_classes": [7, 8],
        "prefix": "scenario-4"
    },
    "5": {
        "inlier_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "outlier_classes": [9],
        "prefix": "scenario-5"
    },
}


class ToTensor(object):
    """Convert data sample (data and labels) to PyTorch tensors."""
    
    def __call__(self, sample):
        image, label = sample["images"], sample["labels"]
        
        # swap color channel axes to `channels first`
        image = image.transpose((2, 0, 1))
        return {"images": torch.from_numpy(image),
                "labels": torch.from_numpy(label)}
    

class ScenarioDatasetMNIST(Dataset):
    """Novelty detection scenario dataset for MNIST."""
    
    def __init__(self, root_dir, partition, scenario, transform=None):
        self.root_dir = root_dir
        self.partition = partition
        self.scenario = scenario
        self.transform = transform
        self.data = np.load(f"{root_dir}/scenario-{scenario}/{partition}_data.npy")
        self.labels = np.load(f"{root_dir}/scenario-{scenario}/{partition}_labels.npy")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        
        sample = {"images": self.data[idx], "labels": self.labels[idx]}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
