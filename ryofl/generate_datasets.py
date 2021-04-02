import click

import tensorflow_federated as tff

from ryofl import common


@click.command()
@click.option(
    '--dataset', help='dataset to prepare',
    type=click.Choice(['femnist', 'cifar100'], case_sensitive=False),
    prompt=True
)
def generate_datasets(dataset):
    """ Download the datasets

    Will save the data in the location specified in `common`
    """

    #  Load the Federated EMNIST dataset, restricted to only digits and only
    #  from clients with more than 2 examples.  Each user's (client) samples
    #  are divided in train and test sets. Full specifications at:
    #  https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data
    print('Generating FEMNIST data')
    tff.simulation.datasets.emnist.load_data(
        only_digits=True,
        cache_dir=common.image_data_dir
    )

    #  Load a federated version of the CIFAR100 dataset. In this case the train
    #  and test sets correspond to a full partition of the clients. More
    #  details at:
    #  https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data
    print('Generating CIFAR100 data')
    tff.simulation.datasets.cifar100.load_data(
        cache_dir=common.image_data_dir
    )


if __name__ == '__main__':
    generate_datasets()

