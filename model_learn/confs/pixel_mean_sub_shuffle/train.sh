#!/usr/bin/env bash
/usr/bin/caffe train -solver ./solver.prototxt -weights ../../models/bvlc_alexnet/bvlc_alexnet.caffemodel 2>&1 | tee ./training.log
../../utils/parse_log.sh ./training.log
python3 ../../utils/plot_learning_curve.py ./training.log ./learning.png