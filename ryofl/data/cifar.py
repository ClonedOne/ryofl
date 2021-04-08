import os

from multiprocessing import Pool

import h5py
import numpy as np

from numpy import ndarray
from sklearn.model_selection import train_test_split

from ryofl import common


def _load_full(base_dir=''):
    """ Load the entire cifar100 dataset

    Args:
        base_dir (str): overwrite default data dir

    Raises:
        FileNotFoundError: Requires the preprocessing script to be run

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

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
        tst_cli = tst_file['examples'][client_id]

        # True labels
        labels_trn = trn_cli['label'][()]

        # Insert channels axis
        matrix_trn = trn_cli['pixels'][()]

        # Accumulate
        trn_x_acc.append(matrix_trn)
        trn_y_acc.append(labels_trn)

        # Cifar train cleints are a superset of the test ones
        if client_id in tst_file['examples']:
            matrix_tst = tst_cli['pixels'][()]
            labels_tst = tst_cli['label'][()]
            tst_x_acc.append(matrix_tst)
            tst_y_acc.append(labels_tst)

    trn_x = np.concatenate(trn_x_acc)
    trn_y = np.concatenate(trn_y_acc)
    tst_x = np.concatenate(tst_x_acc)
    tst_y = np.concatenate(tst_y_acc)

    return trn_x, trn_y, tst_x, tst_y


def _load_single_client(client_id, base_dir=''):
    """ Load the CIFAR100 data for a single client

    Args:
        client_id (str): id of the client's data to load
        base_dir (str): overwrite default data dir

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if not base_dir:
        _dir = common.cifar100_clients_dir
    else:
        _dir = base_dir

    client_f = os.path.join(_dir, '{}_cli-' + str(client_id) + '_{}.npy')

    trn_x_file = client_f.format('trn', 'x')
    if os.path.isfile(trn_x_file):
        trn_x = np.load(trn_x_file, allow_pickle=True)
    else:
        print('WARNING data file {} not found'.format(trn_x_file))
        trn_x = np.array([])

    trn_y_file = client_f.format('trn', 'y')
    if os.path.isfile(trn_y_file):
        trn_y = np.load(trn_y_file, allow_pickle=True)
    else:
        print('WARNING data file {} not found'.format(trn_y_file))
        trn_y = np.array([])

    tst_x_file = client_f.format('tst', 'x')
    if os.path.isfile(tst_x_file):
        tst_x = np.load(tst_x_file, allow_pickle=True)
    else:
        print('WARNING data file {} not found'.format(tst_x_file))
        tst_x = np.array([])

    tst_y_file = client_f.format('tst', 'y')
    if os.path.isfile(tst_y_file):
        tst_y = np.load(tst_y_file, allow_pickle=True)
    else:
        print('WARNING data file {} not found'.format(tst_y_file))
        tst_y = np.array([])

    return trn_x, trn_y, tst_x, tst_y


def _load_data_handler(in_data):
    """ Helper function for multiprocess data loading

    Args:
        in_data (tuple): worker id, list of clients, data directory

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    wid = in_data[0]
    clients = in_data[1]
    _dir = in_data[2]

    # If the clients list is empty, this worker is unused
    if clients.size == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    # Load data for all the clients
    for client_id in clients:
        trn_x, trn_y, tst_x, tst_y = _load_single_client(
            client_id=client_id, base_dir=_dir)

        trn_x_acc.append(trn_x)
        trn_y_acc.append(trn_y)
        tst_x_acc.append(tst_x)
        tst_y_acc.append(tst_y)

    trn_x = np.concatenate([t for t in trn_x_acc if t.size != 0])
    trn_y = np.concatenate([t for t in trn_y_acc if t.size != 0])
    tst_x = np.concatenate([t for t in tst_x_acc if t.size != 0])
    tst_y = np.concatenate([t for t in tst_y_acc if t.size != 0])

    return trn_x, trn_y, tst_x, tst_y


def _load_multi_clients(clients, base_dir=''):
    """ Use multiprocessing to load the data for multiple clients

    Args:
        clients: (list, ndarray, string) ids of the clients to load
        base_dir (str): overwrite default data dir


    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if not base_dir:
        _dir = common.cifar100_clients_dir
    else:
        _dir = base_dir

    workers = common.processors

    cli_lists = np.array_split(clients, workers)
    in_data_l = [(i, cli_lists[i], _dir) for i in range(workers)]

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

    trn_x = np.concatenate([t for t in trn_x_acc if t.size != 0])
    trn_y = np.concatenate([t for t in trn_y_acc if t.size != 0])
    tst_x = np.concatenate([t for t in tst_x_acc if t.size != 0])
    tst_y = np.concatenate([t for t in tst_y_acc if t.size != 0])

    return trn_x, trn_y, tst_x, tst_y


def load_data(clients, frac=1.0):
    """ Load data wrapper for CIFAR100

    Args:
        clients: (list, ndarray, string) ids of the clients to load
        frac (float): fraction of the data to sample

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    #  If the clients parameter is None, return the entire training and test
    #  sets
    if clients is None:
        trn_x, trn_y, tst_x, tst_y = _load_full()

    # If multiple clients are passed, and the number of clients is > number of
    # processes define in the config, use multiprocessing to load the data
    elif isinstance(clients, (ndarray, list)):
        trn_x, trn_y, tst_x, tst_y = _load_multi_clients(clients=clients)

    # Single client id has been passed
    elif isinstance(clients, str):
        trn_x, trn_y, tst_x, tst_y = _load_single_client(client_id=clients)

    else:
        raise NotImplementedError('clients: {} not supported'.format(clients))

    if frac != 1.0:
        print('Sampling factor: ', frac)
        _, trn_x = train_test_split(
            trn_x, test_size=frac, random_state=0, stratify=trn_y)
        _, trn_y = train_test_split(
            trn_y, test_size=frac, random_state=0, stratify=trn_y)
        _, tst_x = train_test_split(
            tst_x, test_size=frac, random_state=0, stratify=tst_y)
        _, tst_y = train_test_split(
            tst_y, test_size=frac, random_state=0, stratify=tst_y)

    return trn_x, trn_y, tst_x, tst_y
