import os
import json

import click
import numpy as np

from ryofl import common
from ryofl.data import utils_data


@click.group()
def cli():
    """ Generate configuration files and scripts
    """


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
    '--clients', help='number of clients participating',
    type=int, default=4
)
@click.option(
    '--rounds', help='number of federated learning rounds',
    type=int, default=10
)
@click.option(
    '--aggregation', help='identifier of federated aggregation function',
    type=str, default='averaging'
)
@click.option(
    '--fraction', help='fraction of the dataset to use',
    type=float, default=1.0
)
@click.option(
    '--epochs', help='number of local training epochs', type=int, default=3
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
    '--min_clients', help='minimum number of clients participating in round',
    type=int, default=0
)
@click.option(
    '--rnd_clients', help='number of clients to sample in round',
    type=int, default=0
)
@click.option(
    '--workers', help='number of worker processes, 0 to use all processors',
    type=int, default=0
)
@click.option(
    '--cliid_list', help='json list of client data files, overrides --clients',
    type=str, default=''
)
@click.option(
    '--no_output', help='set flag to avoid outputs from clients',
    is_flag=True
)
@click.option(
    '--make_script', help='set to generate script to launch clients',
    is_flag=True
)
def make_configs(
        dataset: str,
        model_id: str,
        clients: int,
        rounds: int,
        aggregation: str,
        fraction: float,
        epochs: int,
        batch: int,
        learning_rate: float,
        momentum: float,
        min_clients: int,
        rnd_clients: int,
        workers: int,
        cliid_list: str,
        no_output: bool,
        make_script: bool
):
    # Check if config directory is present, create it if needed
    if not os.path.isdir(common.cfg_dir):
        os.makedirs(common.cfg_dir)

    # If a file list is provided, use the data files listed there
    if cliid_list:
        trn_ids = json.load(
            open(os.path.join(common.saved_files, cliid_list), 'r'))['cliids']
        clients = len(trn_ids)

    # Get the ids of the data chunks and crteate per-client lists
    else:
        trn_ids, _ = utils_data.get_client_ids(dataset, trn=True, tst=False)

    # If not specified, compute min_clients, rnd_clients
    if min_clients == 0:
        min_clients = min(clients, int((clients / 2) + 2))
    if rnd_clients == 0:
        rnd_clients = min(clients, int((clients / 2) + 1))

    trn_ids_x_client = np.array_split(trn_ids, clients)

    # Create configurations
    for i in range(clients + 1):
        cfg_file = os.path.join(common.cfg_dir, 'cfg_file_{}.json'.format(i))

        # All participants configuration
        cfg_dict = {
            'idcli': i,
            'dataset': dataset,
            'model_id': model_id,
            'fraction': fraction,
            'rounds': rounds,
            'epochs': epochs,
            'batch': batch,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'srv_host': common.SRV_HOST,
            'srv_port': common.SRV_PORT,
            'workers': workers
        }

        # Server only configuration
        if i == common.SRV_ID:
            cfg_dict['num_clients'] = clients
            cfg_dict['min_clients'] = min_clients
            cfg_dict['rnd_clients'] = rnd_clients
            cfg_dict['aggregation'] = aggregation

        # If client, also provide list of data chunks
        else:
            cfg_dict['data_clis'] = trn_ids_x_client[i - 1].tolist()
            cfg_dict['no_output'] = no_output

        # Save to file
        json.dump(cfg_dict, open(cfg_file, 'w', encoding='utf-8'), indent=2)

    # If requested, generate a script to run all the clients sequentially
    if make_script:
        client_cmd = 'python -m ryofl.client fl --config configs/cfg_file_{}.json &'
        script_lines = ['#!/bin/bash', ]

        for i in range(1, clients + 1):
            script_lines.append(client_cmd.format(i))
            script_lines.append('sleep 0.1;')

        with open('run_clients.sh', 'w', encoding='utf-8') as script_file:
            for line in script_lines:
                script_file.write(line + '\n')


if __name__ == '__main__':
    cli.add_command(make_configs)
    cli()
