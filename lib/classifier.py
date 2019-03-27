import os
import time
import caffe
import numpy as np
os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output


class Classifier:

    @staticmethod
    def get_net(caffemodel, deploy_file, use_gpu=True):
        """
        Returns an instance of caffe.Net
        :param caffemodel: path to a .caffemodel file
        :param deploy_file: path to a .prototxt file
        :param use_gpu: if True, use the GPU for inference
        :return: caffe model
        """
        if use_gpu:
            caffe.set_mode_gpu()
        try:
            return caffe.Net(deploy_file, caffemodel, caffe.TEST)
        except IOError:
            raise IOError('Cannot load trained caffe model.')

    @staticmethod
    def read_labels(labels_file):
        """
        Returns a list of labels
        :param labels_file: path to a .txt file
        :return: list of labels
        """
        labels = None
        try:
            with open(labels_file) as infile:
                labels = [line.strip() for line in infile]
        except IOError:
            print("Error while reading labels.")
        return labels

    @staticmethod
    def generate_rgb(img_file):
        """
        Generate a 3 channel image matrix
        :param img_file channel image
        :return image matrix with 3 bands by copying content from single band
        """
        return np.stack((np.array(img_file),) * 3, axis=-1)

    @staticmethod
    def classify(caffemodel, deploy_file, image_file, labels_file=None, use_gpu=True, classify_out_que=None):
        """
        Classify one input image with single band using trained caffe model
        :param caffemodel: path to a .caffemodel
        :param deploy_file: path to a .prototxt
        :param image_file: image file
        :param labels_file: class label file
        :param use_gpu: if True, run inference on the GPU
        :param classify_out_que: write classification result in queue
        """
        image_file = Classifier.generate_rgb(image_file)
        net = Classifier.get_net(caffemodel, deploy_file, use_gpu)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.array([104, 117, 123]))
        transformer.set_transpose('data', (2, 0, 1))
        net.blobs['data'].data[...] = transformer.preprocess('data', image_file)
        labels = Classifier.read_labels(labels_file)
        classify_start_time = time.time()
        out = net.forward()
        scores = out['prob'][0]
        print('Classification took %s seconds.' % (time.time() - classify_start_time,))
        indices = (-scores).argsort()[:5]  # take top 5 results
        classifications = []
        for i in indices:
            str_p = '{:9.4%} - "{}"'.format(round(100.0 * scores[i], 4) / 100.0,
                                            labels[i].replace('"', '').split(',')[0])
            classifications.append(str_p)
        classify_out_que.put(classifications)
