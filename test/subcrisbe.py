import random

from paho.mqtt import client as mqtt_client
import logging


broker = '118.69.218.59'
port = 1880
#topic = "aicamera/178b2f0d47581fa0/"
# camera="ff798d71555b9c60"
camera="178b2f0d47581fa0"
topic = "aicamera/{}/".format(camera)
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'


logging.basicConfig(filename="log/{}.log".format(camera),
                    format='%(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def connect_mqtt() -> mqtt_client:
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
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        logger.critical('Received {} from {} topic'.format(msg.payload.decode(),msg.topic))
    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
