import time
import threading
import socketserver


HOST = '127.0.0.1'
PORT = 9999

# Current round of federated learning from the point of view of the server
fl_round_s = 0
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
        server. In this case the server should just acknowledge the operation
        and save the state received from the client.
        """

        global fl_round_s
        global fl_round_s_lock

        data = self.request.recv(1024)
        #  cur_thread = threading.current_thread()
        response = data
        self.request.sendall(response)

        fl_round_s_lock.acquire()
        try:
            fl_round_s += 1
        finally:
            fl_round_s_lock.release()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def serve():
    global fl_round_s
    global fl_round_s_lock
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)

    with server:
        #  ip, port = server.server_address

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        print('Server loop running in thread: {}'.format(server_thread.name))

        while 1:
            time.sleep(1)
            fl_round_s_lock.acquire()
            try:
                print('current value of acc: {}'.format(fl_round_s))
            finally:
                fl_round_s_lock.release()

        server.shutdown()
