
import socket

# AF_INET - family refer to IPv4
# SOCK_STREAM - connection oriented TCP protocol

class TCP_server():

    def __init__(self):
        self.host = ''
        self.port = 1234
        self.client_sock = 0
        self.addr = None
        self.server_sock = None

    def create_Socket(self):
        try: 
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as error:
            print("Socket failed to create!")
            print(f'reason: {error}')

    def bind_socket(self, server_addr, port_no):
        try:
            self.server_sock.bind(('127.0.0.1', 1234))
        except socket.error as error:
            print("Socket failed to bind!")
            print(f'reason: {error}')

    def listen_socket(self, max_allow_conn):
        try:
            self.server_sock.listen(max_allow_conn)
        except socket.error as error:
            print("Socket failed to listen!")
            print(f'reason: {error}')
        
    def init_conn(self):
        try:
            self.client_sock, self.addr = self.server_sock.accept()
            print(f'Client connected from {self.addr}')
        except socket.error as error:
            print("Socket failed to listen!")
            print(f'reason: {error}')
    
    def 

