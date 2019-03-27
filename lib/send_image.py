import pika
import uuid
import numpy as np
import simplejson as json
from PIL import Image


class ImageSender(object):
    """
    RPC for file sending
    """

    def __init__(self, host, queue_name):
        """
        initialized one file sending object
        :param host: rabbitmq server
        :param queue_name: name of queue
        """
        self.response = None
        self.corr_id = None
        self.send_queue_name = queue_name
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(self.on_response, no_ack=False, queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        """
        check correlation id and get response
        """
        if self.corr_id == props.correlation_id:
            self.response = body


class UIImageSender(ImageSender):
    """
    Image sending from UI to classifier
    """

    def call(self, image_f):
        """
        receive image from ui, send it to ocr
        :param image_f: image file
        :return:
        """
        self.corr_id = str(uuid.uuid4())
        # process uploaded image
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
        """
        process image from ui
        :param image_f: image file
        :return: image array as list, and original array shape
        """
        im = Image.open(image_f)
        im = im.convert('L')
        im_arr = np.array(im)
        shape = im_arr.shape
        return im_arr.tolist(), shape


class OCRImageSender(ImageSender):
    """
    Image sending from classifier to ocr
    """

    def call(self, image_f, out_que):
        """
        :param image_f: image file from classifier
        :param out_que: queue to keep output
        """
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
        out_que.put(self.response)


