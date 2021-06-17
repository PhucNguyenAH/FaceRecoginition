import __init__
import pika, sys, os
import json
from time import time

# def main():
#     connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
#     channel = connection.channel()

#     channel.queue_declare(queue='hello')

#     def callback(ch, method, properties, body):
#         print(" [x] Received %r" % body)

#     channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

#     print(' [*] Waiting for messages. To exit press CTRL+C')
#     channel.start_consuming()
count = 0
totaltime = 0
def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        global count
        global totaltime
        my_json = body.decode('utf8').replace("'", '"')
        checktime = time()
        count += 1
        data = json.loads(my_json)
        # print(" [x] Received %r" % data)
        print("Image ", count, "th, Time: ", checktime-data["time"])
        totaltime += checktime-data["time"]
        print("-> Average ", count,"th is ", totaltime/count)
        print()
        
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)