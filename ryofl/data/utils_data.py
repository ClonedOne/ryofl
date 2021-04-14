"""
Generic utilities related to handling the datasets
"""

from typing import Tuple, Any, Iterable

import torch

from numpy import ndarray
from torch.utils.data import TensorDataset, DataLoader

from ryofl import common
from ryofl.data import femnist, cifar


def load_dataset(
    dataset: str,
    clients=None,
    fraction: float = 1.0
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Wrapper function to load data from a dataset

    Can load a single client, multiple clients, or the entire dataset.

    Args:
        dataset (str): identifier of the dataset to load
        clients: (string, list, ndarray) ids of the clients to load
        fraction (float): fraction of the data to sample

    Raises:
        NotImplementedError: dataset must belong to list of supported datasets

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if dataset == 'femnist':
        return femnist.load_data(clients, fraction)

    elif dataset == 'cifar100':
        return cifar.load_data(clients, fraction)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


def get_metadata(dataset: str) -> Tuple[int, int, Any]:
    """ Return dataset specific metadata information

    Args:
        dataset: identifier of the dataset

    Raises:
        NotImplementedError: dataset must belong to list of supported datasets

    Returns
        (int, int, Any): channels, classes, torchvision transformations
    """

    if dataset == 'femnist':
        return femnist.channels, femnist.classes, femnist.transform

    elif dataset == 'cifar100':
        return cifar.channels, cifar.classes, cifar.transform

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


def get_client_ids(dataset: str, trn: bool = True, tst: bool = True) -> Tuple:
    """ Return dataset specific metadata information

    Args:
        dataset: identifier of the dataset
        trn (bool): if true return the list for train ids
        tst (bool): if true return the list for test ids

    Raises:
        NotImplementedError: dataset must belong to list of supported datasets

    Returns
        (int, int, Any): channels, classes, torchvision transformations
    """

    if dataset == 'femnist':
        return femnist.get_client_ids(trn=trn, tst=tst)

    elif dataset == 'cifar100':
        return cifar.get_client_ids(trn=trn, tst=tst)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


def make_dataloader(
    x: Iterable,
    y: Iterable,
    transform: Any,
    shuffle: bool,
    batch: int
) -> DataLoader:
    """ Generate DataLoader from numpy arrays

    Tranform the numpy arrays in torch tensors.
    We also want to apply the transformations defined for each dataset.
    We need to unsqueeze the transformed data points because
    the concatenation will happen over axis 0.

    Args:
        x (Iterable): data matrix
        y (Iterable): labels
        transform (Any): transformations to apply to each point
        shuffle (bool): shuffle the dataset
        batch (int): mini batch size

    Returns:
        DataLoader: torch DataLoader
    """

    tx = torch.cat([torch.unsqueeze(transform(i), 0) for i in x])
    ty = torch.tensor(y)

    ds = TensorDataset(tx, ty)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle,
                    num_workers=common.processors)

    return dl
