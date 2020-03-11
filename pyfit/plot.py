"""
Plotting functions
"""

# Allow uppercase names
# pylint: disable=invalid-name

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyfit import Tensor


def plot_planar_data(X: Tensor, y: Tensor) -> None:
    """Plot some planar data"""

    plt.figure()
    plt.plot(X[y == 0, 0], X[y == 0, 1], "or", alpha=0.5, label=0)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "ob", alpha=0.5, label=1)
    plt.legend()


def plot_decision_boundary(
    pred_func: Any, X: Tensor, y: Tensor, figure: Any = None
) -> None:
    """Plot a decision boundary"""

    if figure is None:  # If no figure is given, create a new one
        plt.figure()
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("Spectral"))
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)


def plot_multiclass(X: Tensor, y: Tensor) -> None:
    """Plot spiral-shaped data"""

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.get_cmap("Set1"), alpha=0.8)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])


def plot_multiclass_decision_boundary(pred_func: Any, X: Tensor, y: Tensor) -> None:
    """Plot a multiclass decision boundary"""

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("tab20b_r"))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.get_cmap("Set1"))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_loss_acc(history: Any) -> None:
    """Plot training and (optionally) validation loss and accuracy"""

    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, ".--", label="Training loss")
    final_loss = loss[-1]
    title = "Training loss: {:.4f}".format(final_loss)
    plt.ylabel("Loss")
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "o-", label="Validation loss")
        final_val_loss = val_loss[-1]
        title += ", Validation loss: {:.4f}".format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history["acc"]

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, ".--", label="Training acc")
    final_acc = acc[-1]
    title = "Training accuracy: {:.2f}%".format(final_acc * 100)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if "val_acc" in history.history:
        val_acc = history.history["val_acc"]
        plt.plot(epochs, val_acc, "o-", label="Validation acc")
        final_val_acc = val_acc[-1]
        title += ", Validation accuracy: {:.2f}%".format(final_val_acc * 100)
    plt.title(title)
    plt.legend()


def plot_loss_mae(history: Any) -> None:
    """Plot training and (optionally) validation loss and MAE"""

    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, ".--", label="Training loss")
    final_loss = loss[-1]
    title = "Training loss: {:.4f}".format(final_loss)
    plt.ylabel("Loss")
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "o-", label="Validation loss")
        final_val_loss = val_loss[-1]
        title += ", Validation loss: {:.4f}".format(final_val_loss)
    plt.title(title)
    plt.legend()

    mae = history.history["mean_absolute_error"]

    plt.subplot(2, 1, 2)
    plt.plot(epochs, mae, ".--", label="Training MAE")
    final_mae = mae[-1]
    title = "Training MAE: {:.2f}".format(final_mae)
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    if "val_mean_absolute_error" in history.history:
        val_mae = history.history["val_mean_absolute_error"]
        plt.plot(epochs, val_mae, "o-", label="Validation MAE")
        final_val_mae = val_mae[-1]
        title += ", Validation MAE: {:.2f}".format(final_val_mae)
    plt.title(title)
    plt.legend()
