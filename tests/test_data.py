import os

from ryofl import common


# Test the existence of dataset files

def test_exists_FEMNIST():
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_emnist_digitsonly_train.h5'))
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets',  'fed_emnist_digitsonly_test.h5'))


def test_exists_CIFAR100():
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_cifar100_train.h5'))
    assert os.path.isfile(os.path.join(
        common.image_data_dir, 'datasets', 'fed_cifar100_test.h5'))

