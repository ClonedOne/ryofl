import os
import h5py

from ryofl import common
from ryofl import generate_datasets
from ryofl.data import femnist


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
