import os

import h5py
import numpy as np

from numpy import ndarray

from ryofl import common


def _load_full():
    """ Load the entire FEMNIST dataset

    Raises:
        FileNotFoundError: Requires the preprocessing script to be run

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    _dir = os.path.join(common.image_data_dir, 'datasets')
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


def load_data(clients, fraction):

    #  If the clients parameter is None, return the entire training and test sets
    if clients is None:
        trn_x, trn_y, tst_x, tst_y = _load_full()
