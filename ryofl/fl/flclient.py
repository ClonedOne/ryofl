import copy
import socket

from ryofl import common
from ryofl.fl import training
from ryofl.data import utils_data
from ryofl.models import utils_model
from ryofl.network import utils_network


def client(cfg: dict):
    """ Federate learning client

    Args:
        cfg (dict): configuration dictionary
    """

    _p = ['{} - {}\n'.format(k, v) for k, v in cfg.items() if k != 'data_clis']
    print('Federated learning client. Config: {}'.format(_p))

    # Unpacking
    idcli = cfg['idcli']
    dataset = cfg['dataset']
    model_id = cfg['model_id']
    fraction = cfg['fraction']
    rounds = cfg['rounds']
    epochs = cfg['epochs']
    batch = cfg['batch']
    learning_rate = cfg['learning_rate']
    momentum = cfg['momentum']
    srv_host = cfg['srv_host']
    srv_port = cfg['srv_port']
    data_clients = cfg['data_clis']
    workers = cfg['workers']
    no_output = cfg['no_output']

    # Load local training data
    trn_x, trn_y, _, _ = utils_data.load_dataset(
        dataset=dataset, clients=data_clients, fraction=fraction, tst=False)
    channels, classes, transform = utils_data.get_metadata(dataset=dataset)
    del _
    print('Training data shapes: {} - {}'.format(trn_x.shape, trn_y.shape))

    # Initialize local model
    local_model = utils_model.build_model(model_id, channels, classes)

    # Initialize loop variables
    fl_round_c = 0
    updated = False

    # Client main loop
    # The client has two main interactions with the server:
    # 1) send the current round number and ask for global state;
    # 2) send the updated local model.
    # Between these two interactions, the client updates its local state.
    while fl_round_c <= rounds:

        # Prepare message
        local_state = copy.deepcopy(local_model.state_dict())
        cli_msg = utils_network.pack_message(
            idc=idcli, fl_r=fl_round_c, upd=updated, m_state=local_state)
        data = b''

        # Send first message and receive reply
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((srv_host, srv_port))
            utils_network.send_message(s, cli_msg)

            data = utils_network.receive_message(s)
        except OSError:
            print('WARNING could not connect to server')

        finally:
            s.close()

        # If communication failed, continue
        if not data:
            continue

        # Unpicle server message
        srv_id, fl_round_s, srv_upd, srv_m_state = utils_network.unpack_message(
            data)

        if srv_id != common.SRV_ID:
            print('WARNING received message from: ', srv_id)
            continue

        # Interaction 1)
        if fl_round_c != fl_round_s and srv_upd:
            fl_round_c = fl_round_s

            # Assign received weights to local model
            local_model.load_state_dict(srv_m_state)

            # Perform local training
            training.train_epochs(
                model=local_model,
                trn_x=trn_x,
                trn_y=trn_y,
                transform=transform,
                epochs=epochs,
                batch=batch,
                lrn_rate=learning_rate,
                momentum=momentum,
                workers=workers,
                no_output=no_output
            )
            updated = True

        # Interaction 2)
        elif fl_round_c == fl_round_s and not srv_upd:
            updated = False

        # Something is wrong with the received message
        else:
            continue


def standalone(
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
        workers (int): number of worker threads
        save_pth (str): path where to save model state
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

    import numpy as np
    np.save('trn_x', trn_x)
    np.save('trn_y', trn_y)

    # Train the model
    training.train_epochs(
        model=model,
        trn_x=trn_x,
        trn_y=trn_y,
        transform=transform,
        epochs=epochs,
        batch=batch,
        lrn_rate=learning_rate,
        momentum=momentum,
        workers=workers
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

    if save_pth:
        utils_model.save_model(model, save_pth)

