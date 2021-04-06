"""
Utility to download the datasets and generate the per-client data matrices.
The _generate* are structured like this to avoid the dependance on
tensorflow_federated in othere modules.
"""

import os

import h5py
import tqdm
import click
import numpy as np
import tensorflow_federated as tff

from ryofl import common


def _generate_femnist(force=False):
    """ Generate the FEMNIST dataset

    Load the Federated EMNIST dataset, restricted to only digits and only
    from clients with more than 2 examples.  Each user's (client) samples
    are divided in train and test sets. Full specifications at:
    https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data

    Args:
        force (bool): force re-generation
    """

    print('Generating FEMNIST data')

    # Download the dataset if not found
    tff.simulation.datasets.emnist.load_data(
        only_digits=True,
        cache_dir=common.image_data_dir
    )

    # Generate the clients' data only if not already present
    if os.path.exists(common.femnist_clients_dir) and not force:
        return

    os.makedirs(common.femnist_clients_dir, exist_ok=True)

    # Open the h5 archives we just downloaded
    train_pth = os.path.join(common.image_data_dir,
                             'datasets', 'fed_emnist_digitsonly_train.h5')
    test_pth = os.path.join(common.image_data_dir, 'datasets',
                            'fed_emnist_digitsonly_test.h5')
    train_file = h5py.File(train_pth, 'r')
    test_file = h5py.File(test_pth, 'r')

    for client_id in tqdm.tqdm(sorted(train_file['examples'])):
        client_f = os.path.join(
            common.femnist_clients_dir, '{}_cli-' + str(client_id) + '_{}')

        # True labels
        labels_trn = train_file['examples'][client_id]['label'][()]
        labels_tst = test_file['examples'][client_id]['label'][()]

        # Insert channels axis
        matrix_trn = np.expand_dims(
            train_file['examples'][client_id]['pixels'][()], axis=-1)
        matrix_tst = np.expand_dims(
            test_file['examples'][client_id]['pixels'][()], axis=-1)

        # Save to file
        np.save(client_f.format('trn', 'x'), matrix_trn)
        np.save(client_f.format('trn', 'y'), labels_trn)
        np.save(client_f.format('tst', 'x'), matrix_tst)
        np.save(client_f.format('tst', 'y'), labels_tst)


def _generate_cifar100(force=False):
    """ Generate the CIFAR100 dataset

    Load a federated version of the CIFAR100 dataset. In this case the train
    and test sets correspond to a full partition of the clients. More
    details at:
    https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data

    Args:
        force (bool): force re-generation
    """

    print('Generating CIFAR100 data')

    # Download the dataset if not found
    tff.simulation.datasets.cifar100.load_data(
        cache_dir=common.image_data_dir
    )

    # Generate the clients' data only if not already present
    if os.path.exists(common.cifar100_clients_dir) and not force:
        return

    os.makedirs(common.cifar100_clients_dir, exist_ok=True)

    # Open the h5 archives we just downloaded
    train_pth = os.path.join(common.image_data_dir,
                             'datasets', 'fed_cifar100_train.h5')
    test_pth = os.path.join(common.image_data_dir, 'datasets',
                            'fed_cifar100_test.h5')
    train_file = h5py.File(train_pth, 'r')
    test_file = h5py.File(test_pth, 'r')

    for client_id in tqdm.tqdm(sorted(train_file['examples'])):
        client_f = os.path.join(
            common.cifar100_clients_dir, '{}_cli-' + str(client_id) + '_{}')

        # True labels
        labels_trn = train_file['examples'][client_id]['label'][()]

        # No need to add channels for cifar100
        matrix_trn = train_file['examples'][client_id]['image'][()]

        # Save to file
        np.save(client_f.format('trn', 'x'), matrix_trn)
        np.save(client_f.format('trn', 'y'), labels_trn)

        # Cifar train cleints are a superset of the test ones
        if client_id in test_file['examples']:
            labels_tst = test_file['examples'][client_id]['label'][()]
            matrix_tst = test_file['examples'][client_id]['image'][()]
            np.save(client_f.format('tst', 'x'), matrix_tst)
            np.save(client_f.format('tst', 'y'), labels_tst)


@click.command()
@click.option(
    '--dataset', help='dataset to prepare',
    type=click.Choice(['femnist', 'cifar100'], case_sensitive=False),
    prompt=True
)
@click.option('--force', is_flag=True, help='force re-generation')
def generate_datasets(dataset, force):
    """ Download the datasets

    Will save the data in the location specified in `common`

    Args:
        dataset (str): dataset to prepare
        force (bool): force re-generation
    """

    if dataset == 'femnist':
        _generate_femnist(force)

    elif dataset == 'cifar100':
        _generate_cifar100(force)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


if __name__ == '__main__':
    generate_datasets()
