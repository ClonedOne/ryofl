import click

from ryofl.fl import training
from ryofl.data import utils_data
from ryofl.network import flclient
from ryofl.models import utils_model


@click.group()
def cli():
    """ Run the federated learning client
    """


@click.command()
def fl():
    flclient.client()


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
@click.option(
    '--batch', help='size of mini batch', type=int, default=32
)
@click.option(
    '--learning_rate', help='optimizer learning rate', type=float, default=0.001
)
@click.option(
    '--momentum', help='optimizer momentum value', type=float, default=0.9
)
def train_standalone(
    dataset: str,
    model_id: str,
    fraction: float,
    epochs: int,
    batch: int,
    learning_rate: float,
    momentum: float
):
    """ Train a standalone model on the dataset

    Will perform normal training on a single client.

    Args:
        dataset (str): identifier of the dataset ot use
        model_id (str): identifier of the model to train
        fraction (float): fraction of the dataset to use
        epochs (int): number of training epochs
        batch (int): size of mini batch
        learning_rate (float): optimizer learning rate
        momentum (float): optimizer momentum value
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
    model = utils_model.build_model(model_id, channels, classes)
    print('Model built:\n', model)

    # Train the model
    training.train_epochs(
        model=model,
        trn_x=trn_x,
        trn_y=trn_y,
        transform=transform,
        epochs=epochs,
        batch=batch,
        lrn_rate=learning_rate,
        momentum=momentum
    )

    # Evaluation
    accuracy = training.eval_model(
        model=model,
        tst_x=tst_x,
        tst_y=tst_y,
        transform=transform,
        batch=batch
    )
    print('Model accuracy on test set: {:.4f}'.format(accuracy))


@click.command()
def train_federated():
    pass


if __name__ == '__main__':
    cli.add_command(train_federated)
    cli.add_command(train_standalone)
    cli.add_command(fl)
    cli()
