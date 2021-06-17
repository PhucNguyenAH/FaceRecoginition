from __init__ import PYTHON_PATH
import pika, sys, os
import json
import pandas as pd
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
time_out = []
average_out = []
size_out = []
def main():
    credentials = pika.PlainCredentials('phuc', '1234')
    connection = pika.BlockingConnection(pika.ConnectionParameters('192.168.1.6',5672,'/',credentials))
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
        # print("Image ", count, "th, Time: ", checktime-data["time"])
        totaltime += checktime-data["time"]
        # print("-> Average ", count,"th is ", totaltime/count)
        # print("Image ", count, "th, Size: ", data["size"])
        # print()
        time_out.append(checktime-data["time"])
        average_out.append(totaltime/count)
        size_out.append(data["size"])
        dictionary = {'time': time_out, 'average': average_out, 'size': size_out}  
        dataframe = pd.DataFrame(dictionary) 
        dataframe.to_csv(os.path.join(PYTHON_PATH,'rabbit.csv'))
        
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