FROM ubuntu:18.04

RUN apt -y update \
    && apt install -y tesseract-ocr libtesseract-dev python3-pip python3-dev python3-setuptools

RUN pip3 install flask flask_cors Pillow pika simplejson numpy pytesseract retrying datefinder

CMD ["/usr/bin/python3", "/ocr/ocr_mq.py"]
