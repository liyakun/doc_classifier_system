#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import os
import time
import caffe
from PIL import Image
import scipy.misc
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output


class Classifier:

    def __init__(self):
        pass

    @staticmethod
    def get_net(caffemodel, deploy_file, use_gpu=True):
        """
        Returns an instance of caffe.Net

        Arguments:
        caffemodel -- path to a .caffemodel file
        deploy_file -- path to a .prototxt file

        Keyword arguments:
        use_gpu -- if True, use the GPU for inference
        """
        if use_gpu:
            caffe.set_mode_gpu()

        # load a new model
        return caffe.Net(deploy_file, caffemodel, caffe.TEST)

    @staticmethod
    def get_transformer(deploy_file):
        """
        Returns an instance of caffe.io.Transformer

        Arguments:
        deploy_file -- path to a .prototxt file

        Keyword arguments:
        mean_file -- path to a .binaryproto file (optional)
        """
        network = caffe_pb2.NetParameter()
        with open(deploy_file) as infile:
            text_format.Merge(infile.read(), network)
        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]
        t = caffe.io.Transformer(inputs={'data': dims})
        t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)

        # color images
        if dims[1] == 3:
            # channel swap
            t.set_channel_swap('data', (2, 1, 0))

        pixel = np.array([104, 117, 123])
        t.set_mean('data', pixel)

        return t

    @staticmethod
    def load_image(path, height, width, mode='RGB'):
        """
        Load an image from disk

        Returns an np.ndarray (channels x width x height)

        Arguments:
        path -- path to an image on disk
        width -- resize dimension
        height -- resize dimension

        Keyword arguments:
        mode -- the PIL mode that the image should be converted to
            (RGB for color or L for grayscale)
        """
        image = Image.open(path)
        image = image.convert(mode)
        image = np.array(image)
        # squash
        image = scipy.misc.imresize(image, (height, width), 'bilinear')
        return image

    @staticmethod
    def forward_pass(images, net, transformer, batch_size=1):
        """
        Returns scores for each image as an np.ndarray (nImages x nClasses)

        Arguments:
        images -- a list of np.ndarrays
        net -- a caffe.Net
        transformer -- a caffe.io.Transformer

        Keyword arguments:
        batch_size -- how many images can be processed at once
            (a high value may result in out-of-memory errors)
        """
        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:, :, np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)

        dims = transformer.inputs['data'][1:]

        scores = None
        for chunk in [caffe_images[x:x + batch_size] for x in range(0, len(caffe_images), batch_size)]:
            new_shape = (len(chunk),) + tuple(dims)
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                image_data = transformer.preprocess('data', image)
                net.blobs['data'].data[index] = image_data
            output = net.forward()[net.outputs[-1]]
            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))
            print('Processed %s/%s images ...' % (len(scores), len(caffe_images)))

        return scores

    @staticmethod
    def read_labels(labels_file):
        """
        Returns a list of strings

        Arguments:
        labels_file -- path to a .txt file
        """
        if not labels_file:
            print('WARNING: No labels file provided. Results will be difficult to interpret.')
            return None

        labels = []
        with open(labels_file) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)
        assert len(labels), 'No labels found'
        return labels

    @staticmethod
    def setup(caffemodel, deploy_file, mean_file=None, use_gpu=False):
        # Load the model and images
        net = Classifier.get_net(caffemodel, deploy_file, use_gpu)
        transformer = Classifier.get_transformer(deploy_file, mean_file)
        _, channels, height, width = transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        return net, transformer, _, channels, height, width, mode

    @staticmethod
    def generate_rgb(img_file):
        size = (227, 227)
        im = Image.open(img_file)
        im = im.convert('L')
        im = im.resize(size)
        return np.stack((np.array(im),) * 3, axis=-1)

    @staticmethod
    def classify(caffemodel, deploy_file, image_file, labels_file=None, use_gpu=True):
        """
        Classify some images against a Caffe model and print the results

        Arguments:
        caffemodel -- path to a .caffemodel
        deploy_file -- path to a .prototxt
        image_files -- list of paths to images

        Keyword arguments:
        mean_file -- path to a .binaryproto
        labels_file path to a .txt file
        use_gpu -- if True, run inference on the GPU
        """
        # Load the model and images
        image_file = Classifier.generate_rgb(image_file)
        net = Classifier.get_net(caffemodel, deploy_file, use_gpu)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.array([104, 117, 123]))
        transformer.set_transpose('data', (2, 0, 1))
        net.blobs['data'].data[...] = transformer.preprocess('data', image_file)
        labels = Classifier.read_labels(labels_file)
        # Classify the image
        classify_start_time = time.time()
        out = net.forward()
        scores = out['prob']
        print('Classification took %s seconds.' % (time.time() - classify_start_time,))
        # Process the results
        indices = (-scores).argsort()[:, :5]  # take top 5 results
        classifications = []
        for image_index, index_list in enumerate(indices):
            result = []
            for i in index_list:
                result.append((labels[i], round(100.0 * scores[image_index, i], 4)))
            classifications.append(result)
        classification_results = ''
        for index, classification in enumerate(classifications):
            for label, confidence in classification:
                classification_results += '{:9.4%} - "{}"'.format(confidence / 100.0, label)
                classification_results += '\n'
        return classification_results
