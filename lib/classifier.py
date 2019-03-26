#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import os
import time
import caffe
import numpy as np

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
    def generate_rgb(img_file):
        """
        Generate a 3 channel image 
        """
        return np.stack((np.array(img_file),) * 3, axis=-1)

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
                classification_results += '{:9.4%} - "{}"\n'.format(confidence / 100.0, label)
        return classification_results
