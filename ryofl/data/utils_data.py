"""
Generic utilities related to handling the datasets
"""

from ryofl.data import femnist, cifar


def load_dataset(dataset, clients=None, fraction=1.0):
    """ Wrapper function to load data from a dataset

    Can load a single client, multiple clients, or the entire dataset.

    Args:
        dataset (str): identifier of the dataset to load
        clients: (string, list, ndarray) ids of the clients to load
        fraction (float): fraction of the data to sample

    Raises:
        NotImplementedError: dataset must belong to list of supported datasets

    Returns:
        (ndarray, ndarray, ndarray, ndarray): trn_x, trn_y, tst_x, tst_y
    """

    if dataset == 'femnist':
        return femnist.load_data(clients, fraction)

    elif dataset == 'cifar100':
        return cifar.load_data(clients, fraction)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

