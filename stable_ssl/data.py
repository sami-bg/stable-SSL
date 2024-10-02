import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from stable_ssl.sampler import PositivePairSampler, ValSampler
import numpy as np

import logging


def load_dataset(dataset_name, data_path, train=True):
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
        return torchvision_dataset(
            root=data_path,
            train=True,
            download=True,
            transform=PositivePairSampler(dataset=dataset_name),
        )

    return torchvision_dataset(
        root=data_path,
        train=False,
        download=True,
        transform=ValSampler(dataset=dataset_name),
    )


def imbalance_torchvision_dataset(
    data_path, dataset, dataset_name, coeff_imbalance=2.0
):
    save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")

    if not os.path.exists(save_path):
        data, labels = from_torchvision(data_path=data_path, dataset=dataset)
        imbalanced_data, imbalanced_labels = resample_classes(
            data, labels, coeff_imbalance=coeff_imbalance
        )
        imbalanced_dataset = {"features": imbalanced_data, "labels": imbalanced_labels}
        save_path = os.path.join(data_path, f"imbalanced_coeff_{coeff_imbalance}.pt")
        torch.save(imbalanced_dataset, save_path)

        print(f"[stable-SSL] Subsampling : imbalanced dataset saved to {save_path}.")

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


def resample_classes(dataset, samples_or_freq, random_seed=None):
    """
    Create an exponential class imbalance.
    Args:
        dataset (torch.utils.data.Dataset): The input data, shape (N, ...).
        samples_or_freq (iterable): Number of samples or frequency
            for each class in the new dataset.
        random_seed (int): The random seed.
    """

    if hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    else:
        raise ValueError("dataset does not have `labels`")
    classes, class_inverse, class_counts = np.unique(
        labels, return_counts=True, return_inverse=True
    )

    logging.info(
        f"[stable-SSL] Subsampling : original class counts: {list(class_counts)}"
    )

    if np.min(samples_or_freq) < 0:
        raise ValueError(
            "You can't have negative values in `sampels_or_freq`, "
            f"got {samples_or_freq}."
        )
    elif np.sum(samples_or_freq) <= 1:
        target_class_counts = np.array(samples_or_freq) * len(dataset)
    elif np.sum(samples_or_freq) == len(dataset):
        freq = np.array(samples_or_freq) / np.sum(samples_or_freq)
        target_class_counts = freq * len(dataset)
        if (target_class_counts / class_counts).max() > 1:
            raise ValueError("specified more samples per class than available")
    else:
        raise ValueError(
            f"samples_or_freq needs to sum to <= 1 or len(datset) ({len(dataset)}), "
            f"got {np.sum(samples_or_freq)}."
        )

    target_class_counts = (
        target_class_counts / (target_class_counts / class_counts).max()
    ).astype(int)

    logging.info(
        f"[stable-SSL] Subsampling : target class counts: {list(target_class_counts)}"
    )

    keep_indices = []
    generator = np.random.Generator(np.random.PCG64(seed=random_seed))
    for cl, count in zip(classes, target_class_counts):
        cl_indices = np.flatnonzero(class_inverse == cl)
        cl_indices = generator.choice(cl_indices, size=count, replace=False)
        keep_indices.extend(cl_indices)
    return torch.utils.data.Subset(dataset, indices=keep_indices)


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
