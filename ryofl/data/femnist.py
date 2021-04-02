import os


def load_client_data(client_id):

    # if the preprocessing has not been run throw eception
    if not os.path.isfile(''):
        raise FileNotFoundError('Run preprocessing script first')
