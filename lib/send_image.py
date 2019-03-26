import pika
import uuid
import numpy as np
import simplejson as json
from PIL import Image


class ImageSender(object):

    def __init__(self, host, queue_name):
        self.response = None
        self.corr_id = None
        self.send_queue_name = queue_name
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(self.on_response, no_ack=False, queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body


class UIImageSender(ImageSender):

    def call(self, image_f):
        self.corr_id = str(uuid.uuid4())
        data, shape = UIImageSender.process_img(image_f=image_f)
        self.channel.basic_publish(exchange='',
                                   routing_key=self.send_queue_name,
                                   properties=pika.BasicProperties(
                                         reply_to=self.callback_queue,
                                         correlation_id=self.corr_id,
                                         ),
                                   body=json.dumps({'data': data, 'shape': shape}))
        while self.response is None:
            self.connection.process_data_events()
        return self.response

    @staticmethod
    def process_img(image_f):
        im = Image.open(image_f)
        im = im.convert('L')
        im_arr = np.array(im)
        shape = im_arr.shape
        return im_arr.tolist(), shape


class OCRImageSender(ImageSender):

    def call(self, image_f):
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.send_queue_name,
                                   properties=pika.BasicProperties(
                                       reply_to=self.callback_queue,
                                       correlation_id=self.corr_id,
                                   ),
                                   body=image_f)
        while self.response is None:
            self.connection.process_data_events()
        return self.response


