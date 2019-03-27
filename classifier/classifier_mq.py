import os
import pika
from retrying import retry
from threading import Thread
import queue
from sys import path
import simplejson as json
import numpy as np
from lib.send_image import OCRImageSender
from lib.classifier import Classifier
from lib.ocr import OCR

CAFFE_MODEL = path[0] + '/models/caffe_alexnet_train_iter_20000.caffemodel'
DEPLOY_FILE = path[0] + '/models/deploy.prototxt'
LABELS_FILE = path[0] + '/models/labels.txt'
RABBIT_HOST = os.environ['RABBIT_HOST']


def image_classifier(ch, method, props, body):
    print('Image received for classification.')
    # start thread for ocr
    ocr_out_que = queue.Queue()
    ocr_thread = Thread(target=OCRImageSender(host=RABBIT_HOST, queue_name='ocr_image').call,
                        args=(body, ocr_out_que))
    ocr_thread.start()
    # reconstruct image
    image = None
    try:
        image = np.array(json.loads(body)['data'], np.uint8).reshape(json.loads(body)['shape'])
    except KeyError:
        raise KeyError('Cannot find image from ui input.')
    # start classification thread
    classify_out_que = queue.Queue()
    classify_thread = Thread(target=Classifier.classify,
                             args=(CAFFE_MODEL, DEPLOY_FILE, image, LABELS_FILE, False, classify_out_que))
    classify_thread.start()
    # collecting result from ocr and classification
    ocr_thread.join()
    classify_thread.join()
    ocr_result = ocr_out_que.get()
    classify_result = classify_out_que.get()
    # extract date or isbn
    extract_type, extract_value = OCR.date_isbn_extraction(classify_result, ocr_result)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=json.dumps({'classify': classify_result,
                                      'ocr': {
                                          'text': ocr_result,
                                          extract_type: extract_value
                                      }}))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(classify_result)


@retry(stop_max_delay=300000)
def get_connection():
    """
    used in docker environment, to prevent RabbitMQ service is not ready while connecting
    :return: connection to rabbitmq server
    """
    return pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST))


if __name__ == '__main__':
    connection = get_connection()
    channel = connection.channel()
    channel.queue_declare(queue='img_classifier')
    channel.basic_consume(image_classifier, queue='img_classifier', no_ack=False)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
