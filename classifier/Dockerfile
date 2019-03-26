FROM nvidia/cuda:10.1-base

RUN apt -y update \
    && apt -y --no-install-recommends install python3-pip caffe-cpu

RUN apt -y install python3-setuptools

RUN pip3 install flask flask_cors Pillow pika simplejson numpy retrying

CMD ["/usr/bin/python3", "/classifier/classifier_mq.py"]
