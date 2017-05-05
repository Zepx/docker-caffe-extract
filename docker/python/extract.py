#!/bin/python

import sys
import os
import numpy as np
caffe_root = os.environ['CAFFE_ROOT'] + '/'

#sys.path.insert(0, caffe_root + 'python')

import caffe
prototxt = sys.argv[1]
caffemodel = sys.argv[2]

if prototxt is None:
    protoxt = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
if caffemodel is None:
    caffemodel = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
for k, v in net.params.items():
    np.save(k, v[0].data)
