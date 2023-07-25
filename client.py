import socket
import sys

# CREATE TCP SOCKET

class TCP_client():

    def __init__(self):
        self.host = ''
        self.port = 1234
        self.sock = None
    
    def create_socket(self):
        try: 
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socke.error as error:
            print("Socket failed to create!")
            print(f'reason: {error}')

    def connect_host(self, host_name, port_no):
        self.host = host_name
        self.port = port_no
        try:
            self.sock.connect((self.host, int(self.port)))
            print(f"connected to {self.host} at {self.port}")
        except socket.error as error:
            print('socket failed to connect!')
            print(f"Reason: {error}")

    def 


client = TCP_client()
client.create_socket()
client.connect_host('www.python.org', 80)
