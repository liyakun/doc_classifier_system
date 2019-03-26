import pika
from sys import path
import simplejson as json
import numpy as np
path.append('../')
from lib.send_image import OCRImageSender
from lib.classifier import Classifier

CAFFE_MODEL = path[0] + '/models/caffe_alexnet_train_iter_20000.caffemodel'
DEPLOY_FILE = path[0] + '/models/deploy.prototxt'
LABELS_FILE = path[0] + '/models/labels.txt'
RABBIT_HOST = 'localhost'


def image_classifier(ch, method, props, body):
    print('Image received for classification.')
    # send image to ocr_image queue
    result = OCRImageSender(host=RABBIT_HOST, queue_name='ocr_image').call(image_f=body)
    image = np.array(json.loads(body)['data'], np.uint8).reshape(json.loads(body)['shape'])
    text = Classifier.classify(caffemodel=CAFFE_MODEL,
                               deploy_file=DEPLOY_FILE,
                               image_file=image,
                               labels_file=LABELS_FILE,
                               use_gpu=True)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=json.dumps({'classify': text, 'ocr': result}))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(text)


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='img_classifier')
channel.basic_consume(image_classifier, queue='img_classifier', no_ack=False)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()