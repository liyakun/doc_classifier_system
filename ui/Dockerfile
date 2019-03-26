FROM ubuntu:18.04

RUN apt -y update \
    && apt -y install python3-pip python3-setuptools

RUN pip3 install flask flask_cors Pillow simplejson flask_wtf pika numpy

CMD ["/usr/bin/python3", "/ui/app.py"]
