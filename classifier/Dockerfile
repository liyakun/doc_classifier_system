FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update \
    && apt install -y tesseract-ocr libtesseract-dev python3-pip caffe-cpu python3-setuptools python3-dev

RUN pip3 install flask flask_cors Pillow pika==0.13.1 simplejson numpy retrying pytesseract datefinder isbnlib

CMD ["/usr/bin/python3", "/classifier/classifier_mq.py"]
