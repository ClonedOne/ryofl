import click

from ryofl import common
from ryofl.data import utils_data
from ryofl.models import utils_model
from ryofl.training import standalone


@click.group()
def cli():
    pass


@click.command()
@click.option(
    '--dataset', help='identifier of the dataset to use',
    type=click.Choice(['femnist', 'cifar100'], case_sensitive=False),
    prompt=True
)
@click.option(
    '--model_id', help='identifier of the model to train',
    type=click.Choice(['cnn', ], case_sensitive=False),
    prompt=True
)
@click.option(
    '--fraction', help='fraction of the dataset to use', type=float, default=1.0
)
@click.option(
    '--epochs', help='number of training epochs', type=int, default=10
)
def train_standalone(dataset: str, model_id: str, fraction: float, epochs: int):
    """ Train a standalone model on the dataset

    Will perform normal training on a single client.

    Args:
        dataset (str): identifier of the dataset ot use
        model_id (str): identifier of the model to train
        fraction (float): fraction of the dataset to use
        epochs (int): number of training epochs
    """

    # Load the dataset
    trn_x, trn_y, tst_x, tst_y = utils_data.load_dataset(
        dataset=dataset, fraction=fraction)
    print(
        'Selected dataset: {}\n'
        '\ttrn_x: {}\n\ttrn_y: {}\n'
        '\ttst_x: {}\n\ttst_y: {}'.format(
            dataset, trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape
        ))
    channels, classes, transform = utils_data.get_metadata(dataset=dataset)

    # Define the model
    cnn = utils_model.build_model(model_id, channels, classes)

    # Train the model
    standalone.standalone_training(cnn, trn_x, trn_y, epochs, transform)


@click.command()
def train_federated():
    pass


if __name__ == '__main__':
    cli.add_command(train_federated)
    cli.add_command(train_standalone)
    cli()
