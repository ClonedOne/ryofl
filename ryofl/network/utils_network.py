import struct


def send_message(s, msg):
    msg_len = len(msg)

    # !Q is network byte order for a long long type
    s.sendall(struct.pack('!Q', msg_len))
    s.sendall(msg)


def receive_message(s):
    # !Q is network byte order for a long long type
    # struct.unpack always returns a tuple
    msg_len = struct.unpack('!Q', s.recv(8))[0]
    print('Incoming message length: {}'.format(msg_len))

    data = b''
    remaining_bytes = msg_len
    while remaining_bytes > 0:
        data += s.recv(msg_len)
        remaining_bytes = msg_len - len(data)

    return data


