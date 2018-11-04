#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:24:33 2018

@author: yonic
"""

import tensorflow as tf
import auto_desc.c3d_model as c3d_model
import auto_desc.utils as utils
import h5py
import numpy as np

def make_model():        
    out = {}
    #TODO:define placeholder here!
    logit,images_placeholder,labels_placeholder = c3d_model.load_trained_model()
    out ['c3d_model_out'] = logit    
    c3d_model_train_vars = tf.trainable_variables()#TODO: set c3d_model_out variables untrainable!
    c3d_model_saver = tf.train.Saver(var_list=c3d_model_train_vars)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))    
    c3d_model_saver.restore(sess, utils.C3D_MODEL)
    print('C3D_MODEL was loaded!')
    
    fd = h5py.File(utils.C3D_MODEL_PCA)
    u = np.array(fd['data']['U'])
    out['pca_dense_matrix'] = pca_dense = tf.convert_to_tensor(u[:,0:utils.C3D_MODEL_PCA_FEATURES],name='pca_dense_matrix')    
    
    with tf.name_scope("c3d_model_out_pca"):
         out['c3d_model_out_pca'] = c3d_model_out_pca = tf.matmul(out['c3d_model_out'],pca_dense)   
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print(out)    
    

if __name__ == '__main__':
    make_model()