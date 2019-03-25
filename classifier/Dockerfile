FROM ubuntu:18.04

RUN apt -y update \
    && apt install python3-pip caffe-cpu caffe-cuda

RUN pip3 install flask flask_cors Pillow

CMD ["/usr/bin/python3", "/classifier/app.py"]