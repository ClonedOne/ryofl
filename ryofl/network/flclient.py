import pickle
import socket

import numpy as np

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
