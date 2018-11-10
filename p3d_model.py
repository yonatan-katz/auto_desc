#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:40:17 2018

@author: yonic
"""

import caffe
import numpy as np
import time

def test():
    net = caffe.Net('models/deploy_p3d_resnet_kinetics.prototxt','models/p3d_resnet_kinetics_iter_190000.caffemodel',caffe.TEST)
    while True:
        t1 = time.time()
        r = net.forward(data=np.random.rand(1,3,16,160,160),end='pool5')
        t2 = time.time()
        print('Sum:',np.sum(r['pool5']),' time:',t2-t1)
    
    

if __name__ == '__main__':
    test()