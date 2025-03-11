"""
This file has the definition of the dataset fashion mnist
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def transform_image(image: torch.Tensor) -> torch.Tensor:
    """flattens the image tensor"""
    image = image.float()  # convert the flattened image to a float tensor to be able to normalize it

    # normalize the image to be between 0 and 1 we devide by 255 because the image is in the range of 0 to 255
    image = image / 255.0
    image = image.unsqueeze(0)  # add a dimension to the tensor to be able to add the channel dimension
    return image


def read_image_torch(image_path: str) -> torch.Tensor:
    """reads the image from the path and returns a tensor"""
    image = Image.open(image_path)
    array_image = np.array(image)
    return torch.as_tensor(array_image)


def read_image_to_labels_file(annotations_file: str, train: bool) -> pd.DataFrame:
    """Reads the image labels from the annotations file"""
    img_labels = pd.read_csv(annotations_file)
    img_labels = img_labels[img_labels["train"] == train]
    return img_labels


class FashionMnistDataset(Dataset):
    """A Pytorch Dataset class for the MNIST dataset"""

    def __init__(self, annotations_file, images_folder, train, transform=None, target_transform=None):
        """_summary_

        Args:
            annotations_file (path): path to the csv file that has the annotations
            images_folder (path): path to the root folder that has train and test folders with images
            train (bool): true for training images and false for test images
            transform (function, optional): the function that transforms images for the model input.
                Defaults to None.
            target_transform (function, optional): the function that transforms the labels for the model input.
                Defaults to None.
        """
        self.img_labels = read_image_to_labels_file(annotations_file, train)
        self.images_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform

    # we always need to define the __len__ method for the dataset
    def __len__(self):
        return len(self.img_labels)

    # we always need to define the __getitem__ method for the dataset
    def __getitem__(self, idx):
        # create path to the image
        img_path = os.path.join(self.images_folder, self.img_labels.iloc[idx, 0])

        # read the image as a torch tensor
        image = read_image_torch(img_path)

        # get the label of the image
        label = self.img_labels.iloc[idx, 1]

        # apply the transformations to the image if they are defined
        if self.transform:
            image = self.transform(image)

        # apply the transformations to the label if they are defined
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
