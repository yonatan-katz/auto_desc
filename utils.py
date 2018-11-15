#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:29:55 2018

@author: yonic
"""
import argparse
import json
import sys
import subprocess
import os

C3D_MODEL =  "./models/c3d_ucf101_finetune_whole_iter_20000_TF.model"
C3D_MODEL_PCA =  "./models/PCA_activitynet_v1-3.hdf5"
C3D_MODEL_PCA_FEATURES = 500 
CAPION_METADATA_FILE = '/home/yonic/repos/auto_desc/dataset/train.json'
VIDEO_ORIG_FOLDER = '/home/yonic/repos/auto_desc/dataset/video/train/'
VIDEO_CONVERTED_FOLDER = '/home/yonic/repos/auto_desc/dataset/video/train/converted/'
BASE_FOLDER = 'https://ai.stanford.edu/~ranjaykrishna/actiongenome/'

def load_caption(caption_fname):
    with open(caption_fname) as fd:
        captions = json.load(fd)
    return captions


def main():
    parser = argparse.ArgumentParser(description='return movie metadata')    
    parser.add_argument('--key', dest='key',  help='movie metadata key')
    args = parser.parse_args()  
    
    if not args.key:
        parser.print_help()
        sys.exit(0)
        
    captions = load_caption(CAPION_METADATA_FILE)
    print('Train keys size:{}'.format(len(captions.keys())))
    scene_meta = captions[args.key]    
    print(json.dumps(scene_meta,indent=2,sort_keys=True))
    
    
    


if __name__ == '__main__':
    main()