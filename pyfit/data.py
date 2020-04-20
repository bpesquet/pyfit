"""
Data utilities

Heavily inspired by https://github.com/joelgrus/joelnet/blob/master/joelnet/data.py
"""

# pylint: disable=too-few-public-methods

import random
from typing import NamedTuple, Iterator, List
from pyfit.engine import Vector

Batch = NamedTuple("Batch", [("inputs", List[Vector]), ("targets", Vector)])


class BatchIterator:
    """Iterates on data by batches"""

    def __init__(
        self,
        inputs: List[Vector],
        targets: Vector,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.inputs: List[Vector] = inputs
        self.targets: Vector = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> Iterator[Batch]:
        starts = list(range(0, len(self.inputs), self.batch_size))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(batch_inputs, batch_targets)
