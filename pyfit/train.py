"""
Training API
"""

# pylint: disable=too-few-public-methods

from typing import Callable, Dict, List
from pyfit.nn import Module
from pyfit.optim import Optimizer
from pyfit.data import BatchIterator
from pyfit.metrics import binary_accuracy

# Used to record training history for metrics
History = Dict[str, List[float]]


class Trainer:
    """Encapsulates the model training loop"""

    def __init__(self, model: Module, optimizer: Optimizer, loss: Callable):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def fit(
        self, data_iterator: BatchIterator, num_epochs: int = 500, verbose: bool = False
    ) -> History:
        """Fits the model to the data"""

        history: History = {"loss": [], "acc": []}
        epoch_loss: float = 0
        epoch_acc: float = 0
        for epoch in range(num_epochs):
            # Reset the gradients of model parameters
            self.optimizer.zero_grad()
            # Reset epoch loss
            epoch_loss = 0

            for batch in data_iterator():
                # Forward pass
                # TODO fix mypy error when mapping model to inputs
                outputs = list(map(self.model, batch.inputs))  # type: ignore

                # Loss computation
                y_pred = [item for sublist in outputs for item in sublist]
                batch_loss = self.loss(batch.targets, y_pred)
                epoch_loss += batch_loss.data

                # Accuracy computation
                # TODO compute epoch accuracy on whole dataset instead of last batch
                epoch_acc = binary_accuracy(batch.targets, y_pred)

                # Backprop and gradient descent
                batch_loss.backward()
                self.optimizer.step()

            # Record training history
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)

            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"loss: {epoch_loss:.6f}, "
                    f"accuracy: {epoch_acc*100:.2f}%"
                )

        return history
