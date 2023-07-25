import cv2
import io
import socket
import struct
import time
import pickle
import numpy as np
import imutils
from datetime import datetime


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('0.tcp.ngrok.io', 19194))
client_socket.connect(('192.168.1.38', 8485))

cam = cv2.VideoCapture(0)
img_counter = 0

#encode to jpeg format
#encode param image quality 0 to 100. default:95
#if you want to shrink data size, choose low image quality.
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=320)
    frame = cv2.flip(frame,180)
    dateTimeObj = datetime.now()
    cv2.putText(frame, str(dateTimeObj), (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)
    result, image = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(image, 0)
    size = len(data)

    if img_counter%10==0:
        client_socket.sendall(struct.pack(">L", size) + data)
        cv2.imshow('client',frame)
        
    img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()