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
# noinspection PyPackageRequirements
import torchvision
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
    os.makedirs(common.image_data_dir, exist_ok=True)

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

    print('Generating CIFAR100/20 data')

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
        labels_trn_coarse = train_file['examples'][client_id]['coarse_label'][()]

        # No need to add channels for cifar100
        matrix_trn = train_file['examples'][client_id]['image'][()]

        # Save to file
        np.save(client_f.format('trn', 'x'), matrix_trn)
        np.save(client_f.format('trn', 'y'), labels_trn)
        np.save(client_f.format('trn', 'y_coarse'), labels_trn_coarse)

        # Cifar train cleints are a superset of the test ones
        if client_id in test_file['examples']:
            labels_tst = test_file['examples'][client_id]['label'][()]
            labels_tst_coarse = test_file['examples'][client_id]['coarse_label'][()]
            matrix_tst = test_file['examples'][client_id]['image'][()]
            np.save(client_f.format('tst', 'x'), matrix_tst)
            np.save(client_f.format('tst', 'y'), labels_tst)
            np.save(client_f.format('tst', 'y_coarse'), labels_tst_coarse)


def _generate_cifar10(force=False):
    """ Generate the CIFAR1011 dataset

    Load a federated version of the CIFAR100 dataset.
    It will download the data form pytorch datasets:
    https://pytorch.org/vision/stable/datasets.html#cifar
    The data will be split among 100 clients so that each client will have
    strictly different data points belonging to exactly 2 classes.

    Args:
        force (bool): force re-generation
    """

    n_clients = 100
    print('Generating CIFAR10/2 data')

    trn = torchvision.datasets.CIFAR10(
        root=common.image_data_dir,
        download=True,
        train=True
    )
    tst = torchvision.datasets.CIFAR10(
        root=common.image_data_dir,
        download=True,
        train=False
    )

    # Generate the clients' data only if not already present
    if os.path.exists(common.cifar10_clients_dir) and not force:
        return
    os.makedirs(common.cifar10_clients_dir, exist_ok=True)

    # Identify classes for 10 --> 2 classes problem transformation
    a_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    a_classes_idx = np.array([trn.class_to_idx[i] for i in a_classes])
    v_classes = ['airplane', 'automobile', 'ship', 'truck']
    v_classes_idx = np.array([trn.class_to_idx[i] for i in v_classes])
    assert np.array_equal(a_classes_idx, [2, 3, 4, 5, 6, 7])
    assert np.array_equal(v_classes_idx, [0, 1, 8, 9])

    # Split the dataset in chunks corresponding to each fine label
    trn_targets = np.array(trn.targets)
    tst_targets = np.array(tst.targets)
    trn_x_split = []
    trn_y_split = []
    tst_x_split = []
    tst_y_split = []

    for c in range(len(trn.classes)):
        trn_subclass_subsets = np.array(np.split(trn.data[trn_targets == c], 20))
        trn_subclass_labels = np.array([[c] * s.shape[0] for s in trn_subclass_subsets])
        trn_x_split.append(trn_subclass_subsets)
        trn_y_split.append(trn_subclass_labels)

        tst_subclass_subsets = np.array(np.split(tst.data[tst_targets == c], 20))
        tst_subclass_labels = np.array([[c] * s.shape[0] for s in tst_subclass_subsets])
        tst_x_split.append(tst_subclass_subsets)
        tst_y_split.append(tst_subclass_labels)

    trn_x_split = np.concatenate(trn_x_split)
    trn_y_split = np.concatenate(trn_y_split)
    tst_x_split = np.concatenate(tst_x_split)
    tst_y_split = np.concatenate(tst_y_split)
    assert trn_x_split.shape == (200, 250, 32, 32, 3)
    assert trn_y_split.shape == (200, 250)
    assert tst_x_split.shape == (200, 50, 32, 32, 3)
    assert tst_y_split.shape == (200, 50)

    # Assign 2 labels to each client
    selected = []
    chunk_ids = set(list(range(trn_x_split.shape[0])))

    for i in range(n_clients):
        c_i_0 = np.random.choice(list(chunk_ids))
        c_y_0 = trn_y_split[c_i_0][0]

        chunk_ids.remove(c_i_0)
        found = False

        while not found:
            c_i_1 = np.random.choice(list(chunk_ids))
            c_y_1 = trn_y_split[c_i_1][0]

            if c_y_0 != c_y_1:
                chunk_ids.remove(c_i_1)
                selected.append((c_i_0, c_i_1))
                found = True

    # Create the clients data file
    for client_id, i in tqdm.tqdm(enumerate(selected)):
        cli_trn_x = np.concatenate([trn_x_split[i[0]], trn_x_split[i[1]]])
        cli_trn_y = np.concatenate([trn_y_split[i[0]], trn_y_split[i[1]]])
        cli_trn_y_b = np.concatenate([
            np.full_like(trn_y_split[i[0]], fill_value=0 if trn_y_split[i[0]][0] in a_classes_idx else 1),
            np.full_like(trn_y_split[i[1]], fill_value=0 if trn_y_split[i[1]][0] in a_classes_idx else 1),
        ])

        cli_tst_x = np.concatenate([tst_x_split[i[0]], tst_x_split[i[1]]])
        cli_tst_y = np.concatenate([tst_y_split[i[0]], tst_y_split[i[1]]])
        cli_tst_y_b = np.concatenate([
            np.full_like(tst_y_split[i[0]], fill_value=0 if tst_y_split[i[0]][0] in a_classes_idx else 1),
            np.full_like(tst_y_split[i[1]], fill_value=0 if tst_y_split[i[1]][0] in a_classes_idx else 1),
        ])

        assert cli_trn_x.shape == (500, 32, 32, 3)
        assert cli_trn_y.shape == (500,)
        assert cli_trn_y_b.shape == (500,)
        assert len(np.unique(cli_trn_y)) == 2
        assert len(np.unique(cli_trn_y_b)) <= 2

        assert cli_tst_x.shape == (100, 32, 32, 3)
        assert cli_tst_y.shape == (100,)
        assert cli_tst_y_b.shape == (100,)
        assert len(np.unique(cli_tst_y)) == 2
        assert len(np.unique(cli_tst_y_b)) <= 2

        client_f = os.path.join(
            common.cifar10_clients_dir, '{}_cli-' + str(client_id) + '_{}')
        np.save(client_f.format('trn', 'x'), cli_trn_x)
        np.save(client_f.format('trn', 'y'), cli_trn_y)
        np.save(client_f.format('trn', 'y_coarse'), cli_trn_y_b)
        np.save(client_f.format('tst', 'x'), cli_tst_x)
        np.save(client_f.format('tst', 'y'), cli_tst_y)
        np.save(client_f.format('tst', 'y_coarse'), cli_tst_y_b)


@click.command()
@click.option(
    '--dataset', help='dataset to prepare',
    type=click.Choice(['femnist', 'cifar100', 'cifar20', 'cifar10', 'cifar2'], case_sensitive=False),
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

    elif dataset == 'cifar20':
        _generate_cifar100(force)

    elif dataset == 'cifar10':
        _generate_cifar10(force)

    elif dataset == 'cifar2':
        _generate_cifar10(force)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))


if __name__ == '__main__':
    generate_datasets()
