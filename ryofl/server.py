import click

from ryofl.fl import flserver


@click.group()
def main():
    """ Run the federated learning server
    """


@click.command()
def fl():
    flserver.serve()


if __name__ == '__main__':
    main.add_command(fl)
    main()
