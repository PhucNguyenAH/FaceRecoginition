import base64
import cv2
import zmq
import os
import imutils
import json
from time import time
from __init__ import PYTHON_PATH

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.bind('tcp://*:5555')


SOURCE = 0
# SOURCE = os.path.join(PYTHON_PATH, "test", "video", "cam1.mp4")
camera = cv2.VideoCapture(SOURCE)  # init the camera

while True:
    grabbed, frame = camera.read()  # grab the current frame
    # frame = imutils.resize(frame, width=min(1920, frame.shape[1]))
    # frame = cv2.resize(frame, (854, 480))  # resize the frame
    # print("frame.shape: ",frame.shape)
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    log = {
            "photo": jpg_as_text,
            "time": time()
        }
    msg = json.dumps(log)
    footage_socket.send_string(msg,zmq.NOBLOCK)
    