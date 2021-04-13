import types
import socket
import selectors

HOST = '127.0.0.1'
PORT = 9999

sel = selectors.DefaultSelector()


def accept_fn(s):
    conn, addr = s.accept()
    print('Connection from: {}'.format(addr))
    conn.setblocking(False)

    data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)


def service_connection(key, mask):

    s = key.fileobj
    data = key.data

    if mask & selectors.EVENT_READ:
        recv_data = s.recv(1024)

        if recv_data:
            data.outb += recv_data

        else:
            print('Closing connection to: {}'.format(data.addr))
            sel.unregister(s)
            s.close()

    if mask & selectors.EVENT_WRITE:
        if data.outb:
            print('echoing: {} - to: {}'.format(repr(data.outb), data.addr))
            sent = s.send(data.outb)
            data.outb = data.outb[sent:]


def serve():

    s = socket.socket(socket.AF_INET,  socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    #  conn, addr = s.accept()

    s.setblocking(False)
    sel.register(s, selectors.EVENT_READ, data=None)


    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_fn(key.fileobj)

            else:
                service_connection(key, mask)

    s.close()

#  def serve():
#
#      s = socket.socket(socket.AF_INET,  socket.SOCK_STREAM)
#      s.bind((HOST, PORT))
#      s.listen()
#      conn, addr = s.accept()
#
#      with conn:
#          print('Connection from: {}'.format(addr))
#
#          while True:
#              data = conn.recv(1024)
#
#              if not data:
#                  break
#
#              conn.sendall(data)
#
#      s.close()

