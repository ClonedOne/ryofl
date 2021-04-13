import pickle
import socket

import numpy as np


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
        'fl_round_c': 12,
        'updated': False,
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
    s.sendall(dmp)

    data = s.recv(1024)
    d = pickle.loads(data)
    print('Received', d)

    s.close()
