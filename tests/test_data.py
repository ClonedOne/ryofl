import os
import h5py
import numpy as np

from ryofl import common
from ryofl import generate_datasets
from ryofl.data import femnist, cifar100


# Test the existence of dataset files

def test_exists_femnist():
    generate_datasets._generate_femnist()
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_emnist_digitsonly_train.h5'))
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets',  'fed_emnist_digitsonly_test.h5'))


def test_exists_cifar100():
    generate_datasets._generate_cifar100()
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_cifar100_train.h5'))
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_cifar100_test.h5'))


def test_load_single_client_femnist():
    random_client = 'f4097_41'

    x, y, tx, ty = femnist._load_single_client(client_id=random_client)

    train_pth = os.path.join(common.image_data_dir,
                             'datasets', 'fed_emnist_digitsonly_train.h5')
    test_pth = os.path.join(common.image_data_dir,
                            'datasets', 'fed_emnist_digitsonly_test.h5')

    train_file = h5py.File(train_pth, 'r')
    test_file = h5py.File(test_pth, 'r')

    images = train_file['examples'][random_client]['pixels']
    labels = train_file['examples'][random_client]['label']
    timages = test_file['examples'][random_client]['pixels']
    tlabels = test_file['examples'][random_client]['label']

    assert x.shape[0] == images.shape[0]
    assert x.shape[1:3] == images.shape[1:]
    assert x.shape[3] == 1
    assert y.shape == labels.shape

    assert tx.shape[0] == timages.shape[0]
    assert tx.shape[1:3] == timages.shape[1:]
    assert tx.shape[3] == 1
    assert ty.shape == tlabels.shape


def test_load_multiple_clients_femnist():
    test_clients = [
        'f0000_14',
        'f0001_41',
        'f0005_26',
        'f0006_12',
        'f0008_45',
        'f0011_13',
        'f0014_19',
        'f0016_39',
        'f0017_07',
        'f0022_10'
    ]

    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    for client_id in test_clients:
        trn_x, trn_y, tst_x, tst_y = femnist._load_single_client(
            client_id=client_id)

        trn_x_acc.append(trn_x)
        trn_y_acc.append(trn_y)
        tst_x_acc.append(tst_x)
        tst_y_acc.append(tst_y)

    trn_x = np.concatenate(trn_x_acc)
    trn_y = np.concatenate(trn_y_acc)
    tst_x = np.concatenate(tst_x_acc)
    tst_y = np.concatenate(tst_y_acc)

    trn_xm, trn_ym, tst_xm, tst_ym = femnist._load_multi_clients(
        clients=test_clients)

    assert np.array_equal(trn_x, trn_xm)
    assert np.array_equal(tst_x, tst_xm)
    assert np.array_equal(trn_y, trn_ym)
    assert np.array_equal(tst_y, tst_ym)


def test_load_multiple_clients_cifar():
    clients = ['99', '100', '101', '102', '103']
    trn_x_acc = []
    trn_y_acc = []
    tst_x_acc = []
    tst_y_acc = []

    train_pth = os.path.join(common.image_data_dir,
                             'datasets', 'fed_cifar100_train.h5')
    test_pth = os.path.join(common.image_data_dir,
                            'datasets', 'fed_cifar100_test.h5')
    trn_file = h5py.File(train_pth, 'r')
    tst_file = h5py.File(test_pth, 'r')

    for client_id in clients:
        trn_cli = trn_file['examples'][client_id]
        labels_trn = trn_cli['label'][()]
        matrix_trn = trn_cli['image'][()]

        # Accumulate
        trn_x_acc.append(matrix_trn)
        trn_y_acc.append(labels_trn)

        # Cifar train cleints are a superset of the test ones
        if client_id in tst_file['examples']:
            tst_cli = tst_file['examples'][client_id]
            matrix_tst = tst_cli['image'][()]
            labels_tst = tst_cli['label'][()]
            tst_x_acc.append(matrix_tst)
            tst_y_acc.append(labels_tst)

    trn_x = np.concatenate(trn_x_acc)
    trn_y = np.concatenate(trn_y_acc)
    tst_x = np.concatenate(tst_x_acc)
    tst_y = np.concatenate(tst_y_acc)

    trn_xm, trn_ym, tst_xm, tst_ym = cifar100._load_multi_clients(clients)

    assert np.array_equal(trn_x, trn_xm)
    assert np.array_equal(tst_x, tst_xm)
    assert np.array_equal(trn_y, trn_ym)
    assert np.array_equal(tst_y, tst_ym)

