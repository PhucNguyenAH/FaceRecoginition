from __init__ import PYTHON_PATH
import pika
import json
import os
import glob
import base64
from time import time
from PIL import Image

# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host='localhost'))
# channel = connection.channel()

# channel.queue_declare(queue='hello')

# channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')
# print(" [x] Sent 'Hello World!'")
# connection.close()

output_path = os.path.join(PYTHON_PATH,"test/output")

# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host='localhost'))
credentials = pika.PlainCredentials('phuc', '1234')
connection = pika.BlockingConnection(
    pika.ConnectionParameters('192.168.1.6',5672, '/',credentials))
channel = connection.channel()

channel.queue_declare(queue='hello')

for filepath in sorted(glob.glob(os.path.join(output_path,'*.jpg')), key=os.path.getmtime):
    print(filepath)
    image_file = Image.open(filepath)
    with open(filepath, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    log = {
            "userId": "phuc",
            "photo": encoded_string,
            "time": time(),
            "size": str(len(image_file.fp.read()))
        }

    msg = json.dumps(log)

    channel.basic_publish(exchange='', routing_key='hello', body=msg)
    print(" [x] Sent json")
connection.close()