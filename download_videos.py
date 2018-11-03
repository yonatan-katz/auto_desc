#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:12:09 2018

@author: yonic
"""
import argparse
import json
import sys
import wget
import os

BASE_FOLDER = 'https://ai.stanford.edu/~ranjaykrishna/actiongenome/'

def main():
    parser = argparse.ArgumentParser(description='Download videos script')        
    parser.add_argument('--id', dest='id', help='json file with videos ids to download')
    parser.add_argument('--out', dest='out', help='folder to save the videos')

    args = parser.parse_args()
    if not args.id or not args.out:
        parser.print_help()
        sys.exit(1)        

    fname = args.id
    out_folder = args.out
    with open(fname,'r') as fd:
        video_ids = json.load(fd)
        for video in video_ids:
            full_fname = os.path.join(out_folder,video+'.mp4')
            if not os.path.exists(full_fname):
                url = os.path.join(BASE_FOLDER,video+'.mp4')
                print('Download {}'.format(url))
                wget.download(url, out=out_folder)
            else:
                print('Skip {}'.format(full_fname))
        
        
if __name__ == '__main__':
    main()