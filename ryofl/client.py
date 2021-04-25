import json
import click

from ryofl.fl import flclient


@click.group()
def cli():
    """ Run the federated learning client
    """


@click.command()
@click.option(
    '--config', help='client configuration file',
    type=str, prompt=True
)
def fl(config):
    cfg = json.load(open(config, 'r', encoding='utf-8'))
    flclient.client(cfg)


@click.command()
@click.option(
    '--dataset', help='identifier of the dataset to use',
    type=str, prompt=True
)
@click.option(
    '--model_id', help='identifier of the model to train',
    type=str, prompt=True
)
@click.option(
    '--fraction', help='fraction of the dataset to use',
    type=float, default=1.0
)
@click.option(
    '--epochs', help='number of training epochs', type=int, default=10
)
@click.option(
    '--batch', help='size of mini batch', type=int, default=32
)
@click.option(
    '--learning_rate', help='optimizer learning rate',
    type=float, default=0.001
)
@click.option(
    '--momentum', help='optimizer momentum value', type=float, default=0.9
)
@click.option(
    '--workers', help='number of worker threads', type=int, default=1
)
@click.option(
    '--save_pth', help='path where to store model state', type=str, default=''
)
def train_standalone(
    dataset: str,
    model_id: str,
    fraction: float,
    epochs: int,
    batch: int,
    learning_rate: float,
    momentum: float,
    workers: int,
    save_pth: str
):
    flclient.standalone(
        dataset,
        model_id,
        fraction,
        epochs,
        batch,
        learning_rate,
        momentum,
        workers,
        save_pth
    )


if __name__ == '__main__':
    cli.add_command(train_standalone)
    cli.add_command(fl)
    cli()
