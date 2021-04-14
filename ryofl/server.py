import os
import json
import click

from ryofl import common
from ryofl.fl import flserver


@click.group()
def main():
    """ Run the federated learning server
    """


@click.command()
@click.option(
    '--config', help='server configuration file',
    type=str, prompt=True
)
def fl(config):
    cfg = json.load(open(config, 'r', encoding='utf-8'))
    flserver.serve(cfg)


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
    '--clients', help='number of clients participating',
    type=int, prompt=True
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
@click.option(
    '--min_clients', help='minimum number of clients participating in round',
    type=int, default=0
)
@click.option(
    '--rnd_clients', help='number of clients to sample in round',
    type=int, default=0
)
def make_configs(
    dataset: str,
    model_id: str,
    clients: int,
    fraction: float,
    epochs: int,
    batch: int,
    learning_rate: float,
    momentum: float,
    min_clients: int,
    rnd_clients: int
):
    # If not specified, compute min_clients, rnd_clients
    if min_clients == 0:
        min_clients = min(clients, int((clients/2) + 2))
    if rnd_clients == 0:
        rnd_clients = min(clients, int((clients/2) + 1))

    # Check if config directory is present, create it if needed
    if not os.path.isdir(common.cfg_dir):
        os.makedirs(common.cfg_dir)

    # Create configurations
    for i in range(clients + 1):
        cfg_file = os.path.join(common.cfg_dir, 'cfg_file_{}.json'.format(i))

        # All participants configuration
        cfg_dict = {
            'idcli': i,
            'dataset': dataset,
            'model_id': model_id,
            'fraction': fraction,
            'epochs': epochs,
            'batch': batch,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'srv_host': common.SRV_HOST,
            'srv_port': common.SRV_PORT
        }

        # Server only configuration
        if i == common.SRV_ID:
            cfg_dict['num_clients'] = clients
            cfg_dict['min_clients'] = min_clients
            cfg_dict['rnd_clients'] = rnd_clients

        # Save to file
        json.dump(cfg_dict, open(cfg_file, 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    main.add_command(fl)
    main.add_command(make_configs)
    main()
