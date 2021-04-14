import copy
import time
import pickle
import threading
import socketserver

from ryofl.data import utils_data
from ryofl.models import utils_model
from ryofl.network import utils_network


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
        data = utils_network.receive_message(self.request)

        # Unpickle the bytes into a dictionary
        data = pickle.loads(data)
        cli_id = data['idcli']
        fl_round_c = data['fl_round']
        updated = data['updated']
        cli_model = data['model_state']

        # Acquire locks to avoid round updates
        fl_round_s_lock.acquire()
        global_state_lock.acquire()
        cli_model_state_lock.acquire()

        try:
            if fl_round_c != fl_round_s:
                # Interaction 1)
                srv_message = utils_network.pack_message(
                    idc=SRV_ID,
                    fl_r=fl_round_s,
                    upd=True,
                    m_state=global_state
                )
                utils_network.send_message(self.request, srv_message)

            elif updated:
                # Interaction 2)
                cli_model_state[cli_id] = cli_model

                # This will act as ack
                srv_message = utils_network.pack_message(
                    idc=SRV_ID,
                    fl_r=fl_round_s,
                    upd=False,
                    m_state={}
                )
                utils_network.send_message(self.request, srv_message)

            else:
                # Something went wrong with the client message, ignore this one
                print('WARNING bad format message: {}'.format(data))

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
    epochs = cfg['epochs']
    batch = cfg['batch']
    learning_rate = cfg['learning_rate']
    momentum = cfg['momentum']
    srv_host = cfg['srv_host']
    srv_port = cfg['srv_port']

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

    # Initialize global model
    global_model = utils_model.build_model(model_id, channels, classes)
    global_state = copy.deepcopy(global_model.state_dict())
    print('Model built:\n', global_state)

    # Initialize server
    server = ThreadedTCPServer((SRV_HOST, SRV_PORT), ThreadedTCPRequestHandler)

    # Server main loop
    with server:

        # Start the server thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print('Server loop running in thread: {}'.format(server_thread.name))

        # Main server loop
        while 1:
            time.sleep(0.1)

            cli_model_state_lock.acquire()
            try:
                print('current value of acc: {}'.format(cli_model_state))
            finally:
                cli_model_state_lock.release()

        # Ensure server thread cleanup
        server.shutdown()
