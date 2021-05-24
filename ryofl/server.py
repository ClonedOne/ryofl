import json

import click

from ryofl.fl import flserver


@click.group()
def main():
    """ Run the federated learning server
    """


def select_server(serv_type: str):
    if serv_type == 'basic':
        return flserver.serve
    else:
        raise NotImplementedError('Server {} not supported'.format(serv_type))


@click.command()
@click.option(
    '--config', help='server configuration file',
    type=str, prompt=True
)
@click.option(
    '-s', '--server_type', help='type of server to instantiate',
    type=str, prompt=True
)
def fl(config, serv_type):
    cfg = json.load(open(config, 'r', encoding='utf-8'))

    serv = select_server(serv_type)
    serv(cfg)


if __name__ == '__main__':
    main.add_command(fl)
    main()
