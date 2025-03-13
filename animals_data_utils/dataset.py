# create a dataset class for the animals dataset

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np


def read_images_to_labels_file(annotations_file: str, train: bool) -> pd.DataFrame:
    """Reads the image labels from the annotations file"""
    img_labels = pd.read_csv(annotations_file)
    img_labels = img_labels[img_labels["train"] == train]
    return img_labels


def read_image_torch(image_path: str) -> torch.Tensor:
    """reads the image from the path and returns a tensor"""
    image = Image.open(image_path)
    array_image = np.array(image)
    tensor_image = torch.as_tensor(array_image).float()
    # change channels order
    tensor_image = tensor_image.permute(2, 0, 1)
    return tensor_image


# load the data from the csv file


class AnimalsDataset(Dataset):
    """A Pytorch Dataset class for the MNIST dataset"""

    def __init__(self, annotations_file, images_folder, train, transform=None, target_transform=None):
        self.img_labels = read_images_to_labels_file(annotations_file, train)
        self.images_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform
        self.class_to_index = {
            "elephant": 0,
            "gorilla": 1,
            "leopard": 2,
            "lion": 3,
            "panda": 4,
            "rhinoceros": 5,
        }
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    # we always need to define the __len__ method for the dataset
    def __len__(self):
        return len(self.img_labels)

    def _get_label_from_index(self, index: int) -> str:
        """Converts an index to a label"""
        return self.index_to_class[index]

    def _get_index_from_label(self, label: str) -> int:
        """Converts a label to an index"""
        return self.class_to_index[label]

    # we always need to define the __getitem__ method for the dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.img_labels.iloc[idx, 0])
        image = read_image_torch(img_path)
        label = self._get_index_from_label(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
