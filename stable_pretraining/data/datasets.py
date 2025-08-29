"""Dataset classes for real data sources.

This module provides dataset wrappers and utilities for working with real data sources
including PyTorch datasets, HuggingFace datasets, and dataset subsets.
"""

import time
from collections.abc import Sequence

import lightning as pl
import torch
from loguru import logger as logging


class Dataset(torch.utils.data.Dataset):
    """Base dataset class with transform support and PyTorch Lightning integration."""

    def __init__(self, transform=None):
        self.transform = transform
        self._trainer = None

    def set_pl_trainer(self, trainer: pl.Trainer):
        self._trainer = trainer

    def process_sample(self, sample):
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("Can't use that keywords")
            if "current_epoch" in sample:
                raise ValueError("Can't use that keywords")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Subset(Dataset):
    r"""Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    @property
    def column_names(self):
        return self.dataset.column_names


class FromTorchDataset(Dataset):
    """Wrapper for PyTorch datasets with custom column naming and transforms.

    Args:
        dataset: PyTorch dataset to wrap
        names: List of names for each element returned by the dataset
        transform: Optional transform to apply to samples
        add_sample_idx: If True, automatically adds 'sample_idx' field to each sample
    """

    def __init__(self, dataset, names, transform=None, add_sample_idx=True):
        super().__init__(transform)
        self.dataset = dataset
        self.names = names
        self.add_sample_idx = add_sample_idx

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {k: v for k, v in zip(self.names, sample)}
        if self.add_sample_idx:
            sample["sample_idx"] = idx
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        columns = list(self.names)
        if self.add_sample_idx and "sample_idx" not in columns:
            columns.append("sample_idx")
        return columns


class HFDataset(Dataset):
    """Hugging Face dataset wrapper with transform and column manipulation support."""

    def __init__(
        self, *args, transform=None, rename_columns=None, remove_columns=None, **kwargs
    ):
        super().__init__(transform)
        import datasets

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            s = int(torch.distributed.get_rank()) * 2
            logging.info(
                f"Sleeping for {s}s to avoid race condition of dataset cache"
                " see https://github.com/huggingface/transformers/issues/15976)"
            )
            time.sleep(s)
        if "storage_options" not in kwargs:
            logging.warning(
                "You didn't pass a storage optionwe are adding one to avoid timeout"
            )
            from aiohttp import ClientTimeout

            kwargs["storage_options"] = {
                "client_kwargs": {"timeout": ClientTimeout(total=3600)}
            }
        dataset = datasets.load_dataset(*args, **kwargs)
        dataset = dataset.add_column("sample_idx", list(range(dataset.num_rows)))
        if rename_columns is not None:
            for k, v in rename_columns.items():
                dataset = dataset.rename_column(k, v)
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)
        self.dataset = dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.process_sample(sample)

    def __len__(self):
        return self.dataset.num_rows

    @property
    def column_names(self):
        return self.dataset.column_names
