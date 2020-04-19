"""
Training API
"""

# pylint: disable=too-few-public-methods

from typing import Callable, Tuple
from pyfit.engine import Scalar
from pyfit.nn import Module
from pyfit.optim import Optimizer
from pyfit.data import BatchIterator
from pyfit.metrics import binary_accuracy


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model: Module, optimizer: Optimizer, loss: Callable):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(
        self, data_iterator: BatchIterator, num_epochs: int = 500, verbose: bool = False
    ) -> Tuple[float, float]:
        """Fits the model to the data"""

        epoch_loss = Scalar(0)
        epoch_acc: float = 0
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            epoch_loss = Scalar(0)

            for batch in data_iterator():
                # Forward pass
                # TODO fix mypy error when mapping model to inputs
                outputs = list(map(self.model, batch.inputs))  # type: ignore

                # Loss computation
                y_pred = [item for sublist in outputs for item in sublist]
                batch_loss = self.loss(batch.targets, y_pred)
                epoch_loss += batch_loss

                # Accuracy computation
                # TODO compute epoch accuracy on whole dataset instead of last batch
                epoch_acc = binary_accuracy(batch.targets, y_pred)

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss.data:.6f}")

        return epoch_loss.data, epoch_acc
