import os

import h5py
import numpy as np

from numpy import ndarray
from sklearn.model_selection import train_test_split

from ryofl import common


def _load_full(base_dir=''):
    """ Load the entire FEMNIST dataset

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

    trn_pth = os.path.join(_dir, 'fed_emnist_digitsonly_train.h5')
    tst_pth = os.path.join(_dir, 'fed_emnist_digitsonly_test.h5')
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
        labels_tst = tst_cli['label'][()]

        # Insert channels axis
        matrix_trn = np.expand_dims(trn_cli['pixels'][()], axis=-1)
        matrix_tst = np.expand_dims(tst_cli['pixels'][()], axis=-1)

        # Accumulate
        trn_x_acc.append(matrix_trn)
        trn_y_acc.append(labels_trn)
        tst_x_acc.append(matrix_tst)
        tst_y_acc.append(labels_tst)

    trn_x = np.concatenate(trn_x_acc)
    trn_y = np.concatenate(trn_y_acc)
    tst_x = np.concatenate(tst_x_acc)
    tst_y = np.concatenate(tst_y_acc)

    return trn_x, trn_y, tst_x, tst_y


def load_single_client(client_id, base_dir=''):
    """ Load the FEMNIST data for a single client

    Args:
        client_id (str): id of the client's data to load
        base_dir (str): overwrite default data dir

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """


    if not base_dir:
        _dir = common.femnist_clients_dir
    else:
        _dir = base_dir

    client_f = os.path.join(_dir, '{}_cli-' + str(client_id) + '_{}')

    trn_x = np.load(client_f.format('trn', 'x'), allow_pickle=True)
    trn_y = np.load(client_f.format('trn', 'y'), allow_pickle=True)
    tst_x = np.load(client_f.format('tst', 'x'), allow_pickle=True)
    tst_y = np.load(client_f.format('tst', 'y'), allow_pickle=True)

    return trn_x, trn_y, tst_x, tst_y



def load_data(clients, frac=1.0):
    """ Load data wrapper for FEMNIST

    Args:
        clients: (list, ndarray, string) ids of the clients to load
        frac: fraction of the data to sample

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    #  If the clients parameter is None, return the entire training and test
    #  sets
    if clients is None:
        trn_x, trn_y, tst_x, tst_y = _load_full()

    # If multiple clients are passed, and the number of clients is > number of
    # processes define in the config, use multiprocessing to load the data
    #  elif isinstance(clients, (ndarray, list)):
    #      pass

    # Single client id has been passed
    elif isinstance(clients, str):
        trn_x, trn_y, tst_x, tst_y = load_single_client(client_id=clients)

    else:
        raise NotImplementedError('clients: {} not supported'.format(clients))

    if frac != 1.0:
        print('Sampling factor: ', frac)
        _, trn_x = train_test_split(trn_x, test_size=frac, random_state=0, stratify=trn_y)
        _, trn_y = train_test_split(trn_y, test_size=frac, random_state=0, stratify=trn_y)
        _, tst_x = train_test_split(tst_x, test_size=frac, random_state=0, stratify=tst_y)
        _, tst_y = train_test_split(tst_y, test_size=frac, random_state=0, stratify=tst_y)

    return trn_x, trn_y, tst_x, tst_y

