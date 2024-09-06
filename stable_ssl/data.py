import os
import numpy as np
import pickle

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from stable_ssl.sampler import PositivePairSampler, ValSampler


def load_dataset(dataset_name, data_path, train=True, coeff_imbalance=None):
    """
    Load a dataset from torchvision.datasets.
    Uses PositivePairSampler for training and ValSampler for validation.
    If coeff_imbalance is not None, create an imbalanced version of the dataset with
    the specified coefficient (exponential imbalance).
    """

    if not hasattr(torchvision.datasets, dataset_name):
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets.")

    torchvision_dataset = getattr(torchvision.datasets, dataset_name)

    if train:
        if coeff_imbalance is not None:
            return imbalance_torchvision_dataset(
                data_path, dataset=torchvision_dataset, coeff_imbalance=coeff_imbalance, dataset_name=dataset_name
            )
        else:
            return torchvision_dataset(
                root=data_path,
                train=True,
                download=True,
                transform=PositivePairSampler(dataset=dataset_name),
            )

    else:
        return torchvision_dataset(
            root=data_path,
            train=False,
            download=True,
            transform=ValSampler(dataset=dataset_name),
        )


def imbalance_torchvision_dataset(data_path, dataset, dataset_name, coeff_imbalance=2.0):
    save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")

    if not os.path.exists(save_path):
        data, labels = from_torchvision(data_path=data_path, dataset=dataset)
        imbalanced_data, imbalanced_labels = create_exponential_imbalance(
            data, labels, coeff_imbalance=coeff_imbalance
        )
        imbalanced_dataset = {"features": imbalanced_data, "labels": imbalanced_labels}
        save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")
        torch.save(imbalanced_dataset, save_path)

        print(f"Imbalanced dataset saved to {save_path}")

    return CustomTorchvisionDataset(
        root=save_path, transform=PositivePairSampler(dataset=dataset_name)
    )


def from_torchvision(data_path, dataset):
    dataset = dataset(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )
    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    return features, labels


def create_exponential_imbalance(data, labels, coeff_imbalance=2.0):
    """
    Create an exponential class imbalance.
    Args:
        data (torch.Tensor): The input data, shape (N, ...).
        labels (torch.Tensor): The class labels, shape (N,).
        coeff_imbalance (float): The imbalance coefficient.
    """
    classes, class_counts = torch.unique(labels, return_counts=True)
    n_classes = len(classes)

    assert torch.all(class_counts == len(labels) / n_classes), "The dataset is not balanced."

    exp_dist = coeff_imbalance ** -torch.arange(n_classes, dtype=torch.float32)
    exp_dist /= exp_dist.max()  # Ensure the first class is not subsampled

    max_samples = class_counts[0].item()
    new_class_counts = (exp_dist * max_samples).to(torch.int32)

    print("New class counts:", new_class_counts)

    keep_indices = []
    for cl, count in zip(classes, new_class_counts):
        cl_indices = torch.nonzero(labels == cl, as_tuple=False).squeeze()
        cl_indices = cl_indices[torch.randperm(len(cl_indices))]  # Shuffle the indices
        keep_indices.append(cl_indices[:count])

    keep_indices = torch.cat(keep_indices)

    return data[keep_indices], labels[keep_indices]


class CustomTorchvisionDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: Directory where the dataset is stored
        transform: A function/transform to apply to the data
        """
        self.transform = transform

        # Load the dataset from the .pt file
        data = torch.load(root)
        self.features = data["features"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = to_pil_image(feature)

        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


if __name__ == "__main__":
    data_path = "../runs/data/CIFAR10/"

    load_dataset("CIFAR10", data_path, train=True, coeff_imbalance=1.1)
