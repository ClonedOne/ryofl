import pickle
import socket

import numpy as np

from ryofl.fl import training
from ryofl.data import utils_data
from ryofl.models import utils_model
from ryofl.network import utils_network


HOST = '127.0.0.1'
PORT = 9999


def client():
    a = np.array([
        [[1.2, 0.5], [2.3, 1.1]],
        [[4.5, 6.7], [7.9, 2.1]],
        [[8.6, 10], [11.1, 3.1]]
    ])
    c = {
        'idcli': 1,
        'fl_round': 0,
        'updated': True,
        'model_state': {
            'size': a.shape,
            'a': a,
            'b': (a - 1)[:2]
        }
    }
    print('sending', c)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.connect((HOST, PORT))
    dmp = pickle.dumps(c)
    #  s.sendall(dmp)
    utils_network.send_message(s, dmp)

    #  data = s.recv(4)
    #  data += s.recv(1000)
    data = utils_network.receive_message(s)
    d = pickle.loads(data)
    print('Received', d)

    s.close()


def standalone(
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
