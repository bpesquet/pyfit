"""
Data utilities

Heavily inspired by https://github.com/joelgrus/joelnet/blob/master/joelnet/data.py
"""

# pylint: disable=too-few-public-methods

import random
from typing import NamedTuple, Iterator
from pyfit.engine import Vector

Batch = NamedTuple("Batch", [("inputs", Vector), ("targets", Vector)])


class BatchIterator:
    """Batch iterator"""

    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Vector, targets: Vector) -> Iterator[Batch]:
        starts = list(range(0, len(inputs), self.batch_size))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
