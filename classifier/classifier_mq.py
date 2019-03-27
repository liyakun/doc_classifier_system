import pika
import os
from retrying import retry
from threading import Thread
import queue
from sys import path
import simplejson as json
import numpy as np
path.append('../')
from lib.send_image import OCRImageSender
from lib.classifier import Classifier

CAFFE_MODEL = path[0] + '/models/caffe_alexnet_train_iter_20000.caffemodel'
DEPLOY_FILE = path[0] + '/models/deploy.prototxt'
LABELS_FILE = path[0] + '/models/labels.txt'
RABBIT_HOST = os.environ['RABBIT_HOST']


def image_classifier(ch, method, props, body):
    print('Image received for classification.')
    # send image to ocr_image queue
    ocr_out_que = queue.Queue()
    ocr_thread = Thread(target=OCRImageSender(host=RABBIT_HOST, queue_name='ocr_image').call,
                        args=(body, ocr_out_que))
    ocr_thread.start()
    image = np.array(json.loads(body)['data'], np.uint8).reshape(json.loads(body)['shape'])
    classify_out_que = queue.Queue()
    classify_thread = Thread(target=Classifier.classify,
                             args=(CAFFE_MODEL, DEPLOY_FILE, image, LABELS_FILE, False, classify_out_que))
    classify_thread.start()
    ocr_thread.join()
    classify_thread.join()
    result = ocr_out_que.get()
    text = classify_out_que.get()
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=json.dumps({'classify': text, 'ocr': result}))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(text)


@retry(stop_max_delay=300000)
def get_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST))


if __name__ == '__main__':
    connection = get_connection()
    channel = connection.channel()
    channel.queue_declare(queue='img_classifier')
    channel.basic_consume(image_classifier, queue='img_classifier', no_ack=False)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
