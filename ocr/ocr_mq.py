import pika
import os
from sys import path
from retrying import retry
import simplejson as json
import numpy as np
path.append('../')
from lib.ocr import OCR

RABBIT_HOST = os.environ['RABBIT_HOST']


def ocr_compute(ch, method, props, body):
    print('Image received in ocr.')
    image = np.array(json.loads(body)['data'], np.uint8).reshape(json.loads(body)['shape'])
    text = OCR.ocr_doc(image_f=image)
    response = text
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(" send ocr result ")


@retry(stop_max_delay=300000)
def get_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST))


if __name__ == '__main__':
    connection = get_connection()
    channel = connection.channel()
    channel.queue_declare(queue='ocr_image')
    channel.basic_consume(ocr_compute, queue='ocr_image', no_ack=False)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
