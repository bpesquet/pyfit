"""
Unit tests for data utilities
"""

# pylint: disable=missing-docstring

from collections import Counter
from typing import List
from pyfit.engine import Vector, Scalar
from pyfit.data import BatchIterator


def test_batch_iterator() -> None:
    # Generate lists of 8 scalars with same values
    inputs: List[Vector] = [[Scalar(x)] for x in range(8)]
    targets: Vector = [Scalar(x) for x in range(8)]

    batch_size = 3
    data_iterator = BatchIterator(inputs, targets, batch_size=batch_size)

    batch_sizes: List[int] = []
    for batch in data_iterator():
        # Check lengths of batches
        assert len(batch.inputs) == len(batch.targets)
        assert len(batch.inputs) <= batch_size
        assert len(batch.targets) <= batch_size

        # Store batch size (same for inputs and targets)
        batch_sizes.append(len(batch.inputs))

    # Check expected sizes of batches (exact order can vary)
    assert Counter(batch_sizes) == Counter([3, 3, 2])

    # TODO check shuffling
