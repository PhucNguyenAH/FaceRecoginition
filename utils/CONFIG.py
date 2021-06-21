import os
from __init__ import PYTHON_PATH
from confluent_kafka import Consumer, KafkaException, Producer
import sys
import getopt
import json
import logging
from pprint import pformat
from time import time
import pandas as pd
import base64
import cv2
import pyrebase
import faiss
import pickle
from dateutil.tz import gettz
from datetime import datetime
import pytz
from time import time
import numpy as np


BOOTSTRAP_SERVER = "192.168.1.6:9092"
GROUP = "None"
TOPIC_CAMERA = "camera1"

CAMERAID = 'cameraID'
STARTTIME = 'startTime'
BASE64 = 'imgSrcBase64'


conf_consumer = {'bootstrap.servers': BOOTSTRAP_SERVER, 'group.id': GROUP, 'session.timeout.ms': 6000,
                'auto.offset.reset': 'smallest', 'fetch.message.max.bytes': 15728640,
                'message.max.bytes': 15728640}
conf_producer = {'bootstrap.servers': BOOTSTRAP_SERVER, 'message.max.bytes': 15728640}

def stats_cb(stats_json_str):
        stats_json = json.loads(stats_json_str)
        print('\nKAFKA Stats: {}\n'.format(pformat(stats_json)))

def get_time():
    return int(time()*1000)

def delivery_callback(err, msg):
    if err:
        sys.stderr.write('%% Message failed delivery: %s\n' % err)
    else:
        sys.stderr.write('%% Message delivered to %s [%d] @ %d\n' %
                            (msg.topic(), msg.partition(), msg.offset()))

class config():
    producer_stream = None
    
    def __init__(self, **kwargs):
        """
        Constructor.
        Args:

        """
        # Create logger for consumer (logs will be emitted when poll() is called)
        logger = logging.getLogger('consumer')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)-15s %(levelname)-8s %(message)s'))
        logger.addHandler(handler)

        # Create Consumer instance
        # Hint: try debug='fetch' to generate some log messages
        config.producer_stream = Producer(**conf_producer)
 

        def print_assignment(consumer, partitions):
            print('Assignment:', partitions)  

    def stream(self, image):
        # Convert captured image to JPG
        retval, buffer = cv2.imencode('.jpg', image)
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        log = {CAMERAID: 1, STARTTIME: time(), BASE64: encoded_string}
        msg = json.dumps(log)
        config.producer_stream.produce(TOPIC_CAMERA, msg, callback=delivery_callback)
        config.producer_stream.poll(0)
        sys.stderr.write('%% Waiting for %d deliveries\n' % len(config.producer_stream))
        config.producer_stream.flush()

    




