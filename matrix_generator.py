#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:21:30 2018

@author: yonic
"""
import argparse
import json
import sys
import wget
import os
import subprocess
import shutil
import auto_desc.utils as utils

CACHE_FOLDER="/tmp/movies"

def download_movie(key):
    url = os.path.join(utils.BASE_FOLDER,key+'.mp4')
    wget.download(url, out=CACHE_FOLDER)

def resample_event_movie(key):
    out_name = '{}/{}.processed'.format(CACHE_FOLDER,key)
    ret_code=subprocess.run(['ffmpeg',
                    '-i','{}/{}.mp4'.format(CACHE_FOLDER,key),                    
                    '-r','24','-s','160x160','-c:v','h264','-b:v','3M','-strict','-2',
                    '-movflags', 'faststart',                    
                    out_name+'~'+'.mp4'],timeout=5).returncode
    if ret_code == 0:
        os.rename(out_name+'~'+'.mp4',out_name+'.mp4')
    
    return ret_code,out_name+'.mp4'

def generate_matrix(key,event_time,event_desc):
    if os.path.exists(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER)
    os.makedirs(CACHE_FOLDER)
    download_movie(key)
    try:
        ret,processed_file = resample_event_movie(key)
    except:
        ret = 1    
    if ret == 0:
        print('Ok')
    else:
        print('Failed')
    pass


if __name__=='__main__':
    generate_matrix('v_zp86ztwZEKk',[],[])
    