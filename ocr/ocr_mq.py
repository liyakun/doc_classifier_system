import pika
import os
import numpy as np
import simplejson as json
from lib.ocr import OCR
from lib.rabbitmq_connect import RabbitServerConnector

RABBIT_HOST = os.environ['RABBIT_HOST']


def ocr_compute(ch, method, props, body):
    """
    call back function return return ocr result
    :param ch: channel
    :param method: meta info for message delivery
    :param props: properties of message
    :param body: content
    """
    text = None
    try:
        image = np.array(json.loads(body)['data'], np.uint8).reshape(json.loads(body)['shape'])
        text = OCR.ocr_doc(image_f=image)
    except KeyError as e:
        print(e)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=text)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(" send ocr result ")


if __name__ == '__main__':
    connection = RabbitServerConnector.get_connection(rabbit_host=RABBIT_HOST)
    channel = connection.channel()
    channel.queue_declare(queue='ocr_image')
    channel.basic_consume(ocr_compute, queue='ocr_image', no_ack=False)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
