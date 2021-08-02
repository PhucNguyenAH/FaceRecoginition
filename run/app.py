from __init__ import PYTHON_PATH
import random
import os
import ssl
import json
import urllib.request as ur
import numpy as np
import cv2
from paho.mqtt import client as mqtt_client
import logging
from time import sleep

from ailibs.detector.retinafacetorch.FaceDetector import FaceDetector


broker = '118.69.218.59'
port = 1880
topicList = ["aicamera/178b2f0d47581fa0/", "aicamera/178b2f0d47581fa0/"]
# broker = 'broker.emqx.io'
# port = 1883
# topicList = ["python/mqtt1", "python/mqtt2"]
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'


logging.basicConfig(filename="{}/logs/{}_{}.log".format(PYTHON_PATH,"phucnah",client_id),
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

result_path = f"{PYTHON_PATH}/logs"

model_path = os.path.join(PYTHON_PATH, "ailibs_data/detector/retinafacetorch/Efficientnet-b2.pth")
NETWORK = "efficientnetb2"
DETECTOR = FaceDetector(detector_model=model_path, network=NETWORK,log=True)

def connect_mqtt(client_id) -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            logger.critical('Connected to MQTT Broker!')
        else:
            print("Failed to connect, return code %d\n", rc)
            logger.critical("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    
    def on_message(client, userdata, msg):
        message = msg.payload.decode()
        print(f"Received `{message}` from `{msg.topic}` topic")
        logger.critical('Received {} from {} topic'.format(message,msg.topic))
        message = json.loads(message)
        sleep(5)
        img_url = message["Image"]
        ssl._create_default_https_context = ssl._create_unverified_context
        s = ur.urlopen(img_url)
        img_raw = np.asarray(bytearray(s.read()), dtype="uint8")
        img_raw = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
        frame = np.float32(img_raw)
        frame_det = frame.copy()
        name = message["StartTime"]
        cv2.imwrite(os.path.join(result_path,f"image/{name}.jpg"),frame)
        dets, score = DETECTOR.detect(frame_det)
        for index, det in enumerate(dets):
            # frameCount += 1
            [left, top, right, bottom] = FaceDetector.get_position(det)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = "{:.4f}".format(score[index])
            cv2.putText(frame, text, (left, top),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imwrite(os.path.join(result_path,f"image_detb2/{name}_facedetection.jpg"),frame)
    
    for topic in topicList:
        client.subscribe(topic)
        client.on_message = on_message

def run():
    print("client_id: ",client_id)
    client = connect_mqtt(client_id)
    subscribe(client)
    client.loop_forever()

if __name__ == '__main__':
    run()
