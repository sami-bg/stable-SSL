"""Utility functions for data manipulation and processing.

This module provides utility functions for working with datasets, including
view folding for contrastive learning and dataset splitting.
"""

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import Optional, Union, cast

import torch
from torch import Generator, default_generator, randperm

from .datasets import Dataset, Subset


def fold_views(tensor, idx):
    """Fold a tensor containing multiple views back into separate views.

    Args:
        tensor: Tensor containing concatenated views
        idx: Sample indices to determine view boundaries

    Returns:
        Tuple of tensors, one for each view
    """
    _, counts = torch.unique_consecutive(idx, return_counts=True)
    if not counts.min().eq(counts.max()):
        raise RuntimeError("counts are not the same for all samples!")
    n_views = counts[0].item()
    fold_shape = (tensor.size(0) // n_views, n_views)
    t = tensor.view(*fold_shape, *tensor.shape[1:])
    return t.unbind(dim=1)


def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> list[Subset]:
    r"""Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
