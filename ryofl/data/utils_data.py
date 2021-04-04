import os
import h5py

from ryofl.data import femnist


def load_dataset(dataset, clients=None, fraction=1.0):

    if dataset == 'femnist':
        return femnist.load_data(clients, fraction)

    elif dataset == 'cifar100':
        #  cifar.load_client_data()
        pass

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

