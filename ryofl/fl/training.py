from typing import Any

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from numpy import ndarray

from ryofl import common
from ryofl.data import utils_data


def train_epochs(
    model: Any,
    trn_x: ndarray,
    trn_y: ndarray,
    transform,
    epochs: int = 10,
    batch: int = 32,
    lrn_rate: float = 0.001,
    momentum: float = 0.9,
    workers: int = 0
):
    """ Train a standalone model on the provided data

    Args:
        model (Any): torch model
        trn_x (ndarray): train data
        trn_y (ndarray): train labels
        transform (Any): torch transformation to apply
        epochs (int): number of epochs
        batch (int): mini batch size
        workers (int): number of training workers
    """

    # Select device to run the computation on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the number of threads to use
    if workers == 0:
        workers = common.processors
    torch.set_num_threads(workers)

    # Create a DataLoader from the given arrays
    trn_dl = utils_data.make_dataloader(
        trn_x, trn_y, transform, shuffle=True, batch=batch, workers=workers)

    # Define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lrn_rate, momentum=momentum)

    # Training loop
    for epoch in range(epochs):

        cur_loss = 0.0
        i = 0
        for i, data in tqdm.tqdm(enumerate(trn_dl, 0)):
            # Send the data to the selected device
            x, y = data[0].to(device), data[1].to(device)

            # The gradients are zeroed out at each epocs to avoid accumulation
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Print loss every 10 mini-batches
            cur_loss += loss.item()

        print('epoch {} - loss: {:.3f}'.format(epoch, cur_loss / i))
        cur_loss = 0.0

        # Evaluation
        accuracy = eval_model(
            model=model,
            tst_x=trn_x,
            tst_y=trn_y,
            transform=transform,
            batch=batch
        )
        print('Model training accuracy: {:.4f}'.format(accuracy))


def eval_model(
    model: Any,
    tst_x: ndarray,
    tst_y: ndarray,
    transform,
    batch: int
) -> float:
    """ Evaluate model on provided data

    Args:
        model Any: pytorch model
        tst_x (ndarray): data matrix
        tst_y (ndarray): labels
        transform: pyth
        batch (int): size of mini batch

    Returns:
        float: accuracy value
    """

    # Accumulators
    correct = 0.0
    total = 0.0

    # Select device to run the computation on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a DataLoader from the given arrays
    tst_dl = utils_data.make_dataloader(
        tst_x, tst_y, transform, shuffle=True, batch=batch)

    # No gradient computation during evaluation
    with torch.no_grad():
        for data in tst_dl:

            # Send the data to the selected device
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)

            # Torch max returns a tuple (value, indices)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy
