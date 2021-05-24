"""
This module deals with handling the CIFAR100 dataset
"""

import os

from typing import Tuple
from multiprocessing import Pool

import h5py
import numpy as np
# noinspection PyPackageRequirements
import torchvision.transforms as transforms

from numpy import ndarray
from sklearn.model_selection import train_test_split

from ryofl import common

# Medata
channels = 3
classes = 100
classes_coarse = 20

# Image transformations
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def _load_full(base_dir: str = '', coarse_labels=False):
    """ Load the entire cifar100 dataset

    Args:
        base_dir (str): overwrite default data dir
        coarse_labels (bool): if true, use cifar100 coarse labels

    Raises:
        FileNotFoundError: Requires the preprocessing script to be run

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    label_indicator = 'label'
    if coarse_labels:
        label_indicator = 'coarse_label'

    if not base_dir:
        _dir = os.path.join(common.image_data_dir, 'datasets')
    else:
        _dir = base_dir

    trn_pth = os.path.join(_dir, 'fed_cifar100_train.h5')
    tst_pth = os.path.join(_dir, 'fed_cifar100_test.h5')
    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    try:
        trn_file = h5py.File(trn_pth, 'r')
    except OSError:
        raise FileNotFoundError(
            '{} missing, run generate_datasets.py'.format(trn_pth))
    try:
        tst_file = h5py.File(tst_pth, 'r')
    except OSError:
        raise FileNotFoundError(
            '{} missing, run generate_datasets.py'.format(tst_pth))

    for client_id in sorted(trn_file['examples']):
        trn_cli = trn_file['examples'][client_id]

        # Extract labels and data
        labels_trn = trn_cli[label_indicator][()]
        matrix_trn = trn_cli['image'][()]

        # Accumulate
        trn_x_acc.append(matrix_trn)
        trn_y_acc.append(labels_trn)

        # Cifar train cleints are a superset of the test ones
        if client_id in tst_file['examples']:
            tst_cli = tst_file['examples'][client_id]
            matrix_tst = tst_cli['image'][()]
            labels_tst = tst_cli[label_indicator][()]
            tst_x_acc.append(matrix_tst)
            tst_y_acc.append(labels_tst)

    trn_x = np.concatenate(trn_x_acc)
    trn_y = np.concatenate(trn_y_acc)
    tst_x = np.concatenate(tst_x_acc)
    tst_y = np.concatenate(tst_y_acc)

    return trn_x, trn_y, tst_x, tst_y


def _load_single_client(client_id, base_dir='', coarse_labels=False):
    """ Load the CIFAR100 data for a single client

    Args:
        client_id (str): id of the client's data to load
        base_dir (str): overwrite default data dir
        coarse_labels (bool): if true, use cifar100 coarse labels

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if not base_dir:
        _dir = common.cifar100_clients_dir
    else:
        _dir = base_dir

    label_indicator = 'y'
    if coarse_labels:
        label_indicator = 'y_coarse'

    client_f = os.path.join(_dir, '{}_cli-' + str(client_id) + '_{}.npy')

    trn_x_file = client_f.format('trn', 'x')
    if os.path.isfile(trn_x_file):
        trn_x = np.load(trn_x_file, allow_pickle=True)
    else:
        trn_x = np.array([])

    trn_y_file = client_f.format('trn', label_indicator)
    if os.path.isfile(trn_y_file):
        trn_y = np.load(trn_y_file, allow_pickle=True)
    else:
        trn_y = np.array([])

    tst_x_file = client_f.format('tst', 'x')
    if os.path.isfile(tst_x_file):
        tst_x = np.load(tst_x_file, allow_pickle=True)
    else:
        tst_x = np.array([])

    tst_y_file = client_f.format('tst', label_indicator)
    if os.path.isfile(tst_y_file):
        tst_y = np.load(tst_y_file, allow_pickle=True)
    else:
        tst_y = np.array([])

    return trn_x, trn_y, tst_x, tst_y


def _load_data_handler(in_data):
    """ Helper function for multiprocess data loading

    Args:
        in_data (tuple): worker id, list of clients, data directory

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    # wid = in_data[0]  # Id of the worker, can be used for debug
    clients = in_data[1]
    _dir = in_data[2]
    coarse_labels = in_data[3]

    # If the clients list is empty, this worker is unused
    if clients.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    # Load data for all the clients
    for client_id in clients:
        trn_x, trn_y, tst_x, tst_y = _load_single_client(
            client_id=client_id, base_dir=_dir, coarse_labels=coarse_labels)

        trn_x_acc.append(trn_x)
        trn_y_acc.append(trn_y)
        tst_x_acc.append(tst_x)
        tst_y_acc.append(tst_y)

    trn_x_acc = [t for t in trn_x_acc if t.size != 0]
    trn_y_acc = [t for t in trn_y_acc if t.size != 0]
    tst_x_acc = [t for t in tst_x_acc if t.size != 0]
    tst_y_acc = [t for t in tst_y_acc if t.size != 0]
    trn_x = np.concatenate(trn_x_acc) if len(trn_x_acc) > 0 else np.array([])
    trn_y = np.concatenate(trn_y_acc) if len(trn_y_acc) > 0 else np.array([])
    tst_x = np.concatenate(tst_x_acc) if len(tst_x_acc) > 0 else np.array([])
    tst_y = np.concatenate(tst_y_acc) if len(tst_y_acc) > 0 else np.array([])

    return trn_x, trn_y, tst_x, tst_y


def _load_multi_clients(clients, base_dir: str = '', workers: int = 0, coarse_labels=False):
    """ Use multiprocessing to load the data for multiple clients

    Args:
        clients: (list, ndarray, string) ids of the clients to load
        base_dir (str): overwrite default data dir
        workers (int): number of dataloader workers
        coarse_labels (bool): if true, use cifar100 coarse labels

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if not base_dir:
        _dir = common.cifar100_clients_dir
    else:
        _dir = base_dir

    if workers == 0:
        workers = common.processors

    cli_lists = np.array_split(clients, workers)
    in_data_l = [(i, cli_lists[i], _dir, coarse_labels) for i in range(workers)]

    with Pool(workers) as p:
        rets = p.map(_load_data_handler, in_data_l)

    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    for ret in rets:
        trn_x_acc.append(ret[0])
        trn_y_acc.append(ret[1])
        tst_x_acc.append(ret[2])
        tst_y_acc.append(ret[3])

    trn_x_acc = [t for t in trn_x_acc if t.size != 0]
    trn_y_acc = [t for t in trn_y_acc if t.size != 0]
    tst_x_acc = [t for t in tst_x_acc if t.size != 0]
    tst_y_acc = [t for t in tst_y_acc if t.size != 0]
    trn_x = np.concatenate(trn_x_acc) if len(trn_x_acc) > 0 else np.array([])
    trn_y = np.concatenate(trn_y_acc) if len(trn_y_acc) > 0 else np.array([])
    tst_x = np.concatenate(tst_x_acc) if len(tst_x_acc) > 0 else np.array([])
    tst_y = np.concatenate(tst_y_acc) if len(tst_y_acc) > 0 else np.array([])

    return trn_x, trn_y, tst_x, tst_y


def load_data(
        clients,
        frac: float = 1.0,
        tst: bool = True,
        coarse_labels: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Load data wrapper for CIFAR100

    Args:
        clients: (list, ndarray, string) ids of the clients to load
        frac (float): fraction of the data to sample
        tst (bool): if true load test data
        coarse_labels (bool): if true, use cifar100 coarse labels

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    #  If the clients parameter is None, return the entire training and test
    #  sets
    if clients is None:
        trn_x, trn_y, tst_x, tst_y = _load_full(coarse_labels=coarse_labels)

    # If multiple clients are passed, and the number of clients is > number of
    # processes define in the config, use multiprocessing to load the data
    elif isinstance(clients, (ndarray, list)):
        trn_x, trn_y, tst_x, tst_y = _load_multi_clients(clients=clients, coarse_labels=coarse_labels)

    # Single client id has been passed
    elif isinstance(clients, str):
        trn_x, trn_y, tst_x, tst_y = _load_single_client(client_id=clients, coarse_labels=coarse_labels)

    else:
        raise NotImplementedError('clients: {} not supported'.format(clients))

    if frac != 1.0:
        print('Sampling factor: ', frac)
        _, trn_x = train_test_split(
            trn_x, test_size=frac, random_state=0, stratify=trn_y)
        _, trn_y = train_test_split(
            trn_y, test_size=frac, random_state=0, stratify=trn_y)

        if tst:
            _, tst_x = train_test_split(
                tst_x, test_size=frac, random_state=0, stratify=tst_y)
            _, tst_y = train_test_split(
                tst_y, test_size=frac, random_state=0, stratify=tst_y)
        else:
            tst_x = np.array([])
            tst_y = np.array([])

    return trn_x, trn_y, tst_x, tst_y


def get_client_ids(trn: bool = True, tst: bool = True, base_dir: str = '') -> Tuple:
    """ Return list of client ids for training and test sets

    Args:
        trn (bool): if true return the list for train ids
        tst (bool): if true return the list for test ids
        base_dir (str): overwrite default data dir

    Returns:
        Tuple: train ids, test ids
    """

    trn_ids = []
    tst_ids = []

    if not base_dir:
        _dir = common.cifar100_clients_dir
    else:
        _dir = base_dir

    if trn:
        trn_ids = sorted(set(
            [i.split('-')[1].split('_')[0] for i in os.listdir(_dir) if 'trn' in i]
        ))

    if tst:
        tst_ids = sorted(set(
            [i.split('-')[1].split('_')[0] for i in os.listdir(_dir) if 'tst' in i]
        ))

    return trn_ids, tst_ids
