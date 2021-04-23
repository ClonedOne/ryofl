import copy
import time
import threading
import socketserver

import numpy as np

from ryofl.data import utils_data
from ryofl.models import utils_model
from ryofl.network import utils_network
from ryofl.fl import aggregations, training


# Current round of federated learning from the point of view of the server
fl_round_s = 1
fl_round_s_lock = threading.Lock()

# Dictionary with state of global model
global_state = {}
global_state_lock = threading.Lock()

# dictionary of received client models
cli_model_state = {}
cli_model_state_lock = threading.Lock()


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        """ Handle incoming connections form clients

        There are two distinct steps when a message may arrive from the
        clients.

        1) The client has just started the current round and is polling to get
        the state of the global model and the current logical time stamp. In
        this case the server should respond by sending its state. The state
        received from the client can be discarded.

        2) The client has computed the local update and is sending it to the
        server. In this case the server should just acknowledge reception and
        save the state received from the client.

        Messages from clients have the form:
        data = {
            'idcli': int,
            'fl_round': int,
            'updated': bool,
            'model_state': dict
        }
        """

        # Global variables and locks that will be accessed
        global SRV_ID
        global fl_round_s, fl_round_s_lock
        global global_state, global_state_lock
        global cli_model_state, cli_model_state_lock

        # Receive the message bytes
        data = b''
        try:
            data = utils_network.receive_message(self.request)
        except OSError:
            print('WARNING problems with request:', self.request)

        finally:
            # If the first connection failed, just skip interaction
            if not data:
                return

        # Unpickle the bytes into a dictionary
        cli_id, fl_round_c, updated, cli_model = utils_network.unpack_message(data)

        # Acquire locks to avoid round updates
        fl_round_s_lock.acquire()
        global_state_lock.acquire()
        cli_model_state_lock.acquire()

        try:
            # If something goes wrong with the client message, the default
            # message is ack
            srv_upd = False
            srv_state = {}

            if fl_round_c != fl_round_s:
                # Interaction 1)
                srv_upd = True
                srv_state = copy.deepcopy(global_state)

            elif updated:
                # Interaction 2)
                # Response will act as ack
                cli_model_state[cli_id] = cli_model

            srv_message = utils_network.pack_message(
                idc=SRV_ID,
                fl_r=fl_round_s,
                upd=srv_upd,
                m_state=srv_state
            )
            utils_network.send_message(self.request, srv_message)

        except OSError:
            print('WARNING problems with communications with client', cli_id)

        finally:
            # Release all locks
            fl_round_s_lock.release()
            global_state_lock.release()
            cli_model_state_lock.release()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def serve(cfg: dict):
    """ Federated learning server

    This server will act as aggregator for the updates coming from each client.
    The expected behavior is:
    - if the client is connecting for the first time in this round, send the current global state;
    - if the client is connecting for the second time, store the state of the client model;
    - when enough clients have sent their state, aggregate them and update the global model;
    - evaluate current state of global model on test set.

    Args:
        cfg (dict): configuration dictionary
    """

    print('Starting federated learning server. Received config: {}'.format(cfg))

    # Unpacking
    dataset = cfg['dataset']
    model_id = cfg['model_id']
    fraction = cfg['fraction']
    rounds = cfg['rounds']
    batch = cfg['batch']
    aggregation = cfg['aggregation']

    # Global configuration values
    global SRV_ID, SRV_HOST, SRV_PORT, NUMCLIENTS, MINCLIENTS, RNDCLIENTS
    SRV_ID = cfg['idcli']
    SRV_HOST = cfg['srv_host']
    SRV_PORT = cfg['srv_port']
    NUMCLIENTS = cfg['num_clients']
    MINCLIENTS = cfg['min_clients']
    RNDCLIENTS = cfg['rnd_clients']

    # Global variables and locks that will be accessed
    global fl_round_s, fl_round_s_lock
    global global_state, global_state_lock
    global cli_model_state, cli_model_state_lock

    # Load test data
    _, _, tst_x, tst_y = utils_data.load_dataset(
        dataset=dataset, clients=None, fraction=fraction)
    channels, classes, transform = utils_data.get_metadata(dataset=dataset)
    del _
    print('Testing data shapes: {} - {}'.format(tst_x.shape, tst_y.shape))

    # Initialize global model
    global_model = utils_model.build_model(model_id, channels, classes)
    global_state = copy.deepcopy(global_model.state_dict())
    print('Model built:\n', global_model)

    # Initialize server
    server = ThreadedTCPServer((SRV_HOST, SRV_PORT), ThreadedTCPRequestHandler)
    accuracies = []

    # Server main loop
    with server:

        # Start the server thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print('Server loop running in thread: {}'.format(server_thread.name))

        # Main server loop
        while fl_round_s <= rounds:
            time.sleep(0.1)

            # Check the number of received models
            cli_model_state_lock.acquire()

            try:
                n_received_models = len(cli_model_state)

            finally:
                cli_model_state_lock.release()

            # If we don't have enough models, just continue listening
            if n_received_models < MINCLIENTS:
                continue

            # Otherwise:
            # - acquire all locks, so that clents won't proceed
            # - sample a subset of clients to use in aggregation
            # - aggregate weights
            # - test current state of the model
            # - update fl_round_s number

            fl_round_s_lock.acquire()
            global_state_lock.acquire()
            cli_model_state_lock.acquire()

            try:
                # Sample clients for the current round
                rnd_clis = np.random.choice(
                    list(cli_model_state.keys()), size=RNDCLIENTS, replace=False
                )
                rnd_cli_weights = copy.deepcopy(
                    [cli_model_state[i] for i in rnd_clis])
                rnd_cli_weights = [global_state, ] + rnd_cli_weights

                # Aggregate weights and update global model
                rnd_weights = aggregations.aggregate(
                    rnd_cli_weights,
                    aggregation
                )
                global_model.load_state_dict(rnd_weights)

                # Evaluate global model
                rnd_acc = training.eval_model(
                    model=global_model,
                    tst_x=tst_x,
                    tst_y=tst_y,
                    transform=transform,
                    batch=batch
                )
                accuracies.append(rnd_acc)
                print('Global model accuracy at round {}: {:.5f}'.format(
                    fl_round_s, rnd_acc))

                # Finally update the state of global variables
                global_state = rnd_weights
                cli_model_state = {}
                fl_round_s += 1

            finally:
                fl_round_s_lock.release()
                global_state_lock.release()
                cli_model_state_lock.release()

        # Ensure server thread cleanup
        server.shutdown()
