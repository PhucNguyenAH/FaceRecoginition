import cv2
import zmq
import base64
import numpy as np
import json
import os
import pandas as pd 
from time import time
from __init__ import PYTHON_PATH

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.connect('tcp://localhost:5555')

footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

poller = zmq.Poller()
poller.register(footage_socket, zmq.POLLIN)

class Streamer():

    def __init__(self, **kwargs):
        self.totaltime = 0
        self.totaltimeFrame = 0
        self.count = 0
        self.time_out = []
        self.average_out = []
        self.timeFrame = []
        self.averageFrame_out = []
        self.oldtime = time()

    def streamer(self):
        while True:
            socks = dict(poller.poll(100))
            if socks:
                if socks.get(footage_socket) == zmq.POLLIN:
                    msg = footage_socket.recv_string(zmq.NOBLOCK)
                    # my_json = msg.endode().decode('utf8').replace("'", '"')
                    data = json.loads(msg)
                    img = base64.b64decode(data['photo'])
                    npimg = np.fromstring(img, dtype=np.uint8)
                    source = cv2.imdecode(npimg, 1)
                    checktime = time()
                    self.count += 1
                    self.time_out.append(checktime-data["time"])
                    self.timeFrame.append(checktime-self.oldtime)
                    self.totaltimeFrame += checktime-self.oldtime
                    self.oldtime = checktime
                    self.totaltime += checktime-data["time"]
                    self.average_out.append(self.totaltime/self.count)
                    self.averageFrame_out.append(self.totaltimeFrame/self.count)
                    dictionary = {'time': self.time_out, 'average': self.average_out, 
                                'timeFrame': self.timeFrame, 'avgFrame': self.averageFrame_out}
                    # print("Time: ", time() - data['time'])
                    dataframe = pd.DataFrame(dictionary) 
                    dataframe.to_csv(os.path.join(PYTHON_PATH,'ZMQVideo1920x1080poll.csv'))
                    return cv2.imencode('.jpg',source)[1].tobytes()

            # source.tobytes()

