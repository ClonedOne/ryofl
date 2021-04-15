"""
General utilities for communications over the network
"""

import struct
import pickle

from typing import Any, Tuple


def send_message(s: Any, msg: bytes):
    """ Send a message over the network

    Uses an open socket to send a message encoded as a byte array.

    Args:
        s (Any): open socket
        msg (bytes): byte array to send, could be a numpy array
    """

    msg_len = len(msg)

    # !Q is network byte order for a long long type
    s.sendall(struct.pack('!Q', msg_len))
    s.sendall(msg)


def receive_message(s: Any) -> bytes:
    """ Receive a message from the network

    Uses an open socket ot receive a message

    Args:
        s (Any): open socket

    Returns:
        bytes: received bytes
    """

    # !Q is network byte order for a long long type
    # struct.unpack always returns a tuple
    msg_len = struct.unpack('!Q', s.recv(8))[0]

    data = b''
    remaining_bytes = msg_len
    while remaining_bytes > 0:
        data += s.recv(msg_len)
        remaining_bytes = msg_len - len(data)

    return data


def pack_message(idc: int, fl_r: int, upd: bool, m_state: dict) -> bytes:
    """ Create the data message bytes

    Messages from clients have the form:
        data = {
            'idcli': int,
            'fl_round': int,
            'updated': bool,
            'model_state': dict
        }

    Args:
        idc (int): identifier of the participant
        fl_r (int): round number
        upd (bool): model update flag, used by clients
        m_state (dict): state of the model

    Returns:
        bytes: message bytes
    """

    data = {
        'idcli': idc,
        'fl_round': fl_r,
        'updated': upd,
        'model_state': m_state
    }

    data_b = pickle.dumps(data)
    return data_b


def unpack_message(data_r: bytes) -> Tuple:
    """ Read the message bytes

    Messages from clients have the form:
        data = {
            'idcli': int,
            'fl_round': int,
            'updated': bool,
            'model_state': dict
        }

    Args:
        data (bytes): message bytes
        fl_r (int): round number
        upd (bool): model update flag, used by clients
        m_state (dict): state of the model

    Returns:
        Tuple: idcli, fl_round, updated, model_state
    """

    data: dict = pickle.loads(data_r)
    cli_id = data['idcli']
    fl_round = data['fl_round']
    updated = data['updated']
    m_state = data['model_state']

    return cli_id, fl_round, updated, m_state

