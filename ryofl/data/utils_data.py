"""
Generic utilities related to handling the datasets
"""

from typing import Tuple, Any, Iterable

# noinspection PyPackageRequirements
import torch

from numpy import ndarray
# noinspection PyPackageRequirements
from torch.utils.data import TensorDataset, DataLoader

from ryofl import common
from ryofl.data import femnist, cifar100, cifar10


def load_dataset(
        dataset: str,
        clients=None,
        fraction: float = 1.0,
        tst: bool = True,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Wrapper function to load data from a dataset

    Can load a single client, multiple clients, or the entire dataset.

    Args:
        dataset (str): identifier of the dataset to load
        clients: (string, list, ndarray) ids of the clients to load
        fraction (float): fraction of the data to sample
        tst (bool): if true load test data

    Raises:
        NotImplementedError: dataset must belong to list of supported datasets

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if dataset == 'femnist':
        return femnist.load_data(clients, fraction, tst)

    elif dataset == 'cifar100':
        return cifar100.load_data(clients, fraction, tst)

    elif dataset == 'cifar20':
        return cifar100.load_data(clients, fraction, tst, coarse_labels=True)

    elif dataset == 'cifar10':
        return cifar10.load_data(clients, fraction, tst, coarse_labels=False)

    elif dataset == 'cifar2':
        return cifar10.load_data(clients, fraction, tst, coarse_labels=True)

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
        return cifar100.channels, cifar100.classes, cifar100.transform

    elif dataset == 'cifar20':
        return cifar100.channels, cifar100.classes_coarse, cifar100.transform

    elif dataset == 'cifar10':
        return cifar10.channels, cifar10.classes, cifar10.transform

    elif dataset == 'cifar2':
        return cifar10.channels, cifar10.classes_coarse, cifar10.transform

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
        return cifar100.get_client_ids(trn=trn, tst=tst)

    elif dataset == 'cifar20':
        return cifar100.get_client_ids(trn=trn, tst=tst)

    elif dataset == 'cifar10':
        return cifar10.get_client_ids(trn=trn, tst=tst)

    elif dataset == 'cifar2':
        return cifar10.get_client_ids(trn=trn, tst=tst)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


def make_dataloader(
        x: Iterable,
        y: Iterable,
        transform: Any,
        shuffle: bool,
        batch: int,
        workers: int = 0
) -> DataLoader:
    """ Generate DataLoader from numpy arrays

    Transform the numpy arrays in torch tensors.
    We also want to apply the transformations defined for each dataset.
    We need to unsqueeze the transformed data points because
    the concatenation will happen over axis 0.

    Args:
        x (Iterable): data matrix
        y (Iterable): labels
        transform (Any): transformations to apply to each point
        shuffle (bool): shuffle the dataset
        batch (int): mini batch size
        workers (int): number of dataloader workers

    Returns:
        DataLoader: torch DataLoader
    """

    if workers == 0:
        workers = common.processors

    tx = torch.cat([torch.unsqueeze(transform(i), 0) for i in x])
    ty = torch.tensor(y)

    ds = TensorDataset(tx, ty)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers)

    return dl
