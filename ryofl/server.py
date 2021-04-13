import click

from ryofl.network.wsgi import app
from ryofl.network import flserver


@click.group()
def main():
    """ Run the federated learning server
    """


@click.command()
def web():
    app.run()


@click.command()
def fl():
    flserver.serve()


if __name__ == '__main__':
    main.add_command(web)
    main.add_command(fl)
    main()


