#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:25:07 2018

@author: yonic
"""
import numpy as np
import cv2
import glob
import os
import time
import auto_desc.utils as utils


'''Load video scene accordnig to video file name, 
   start and end are given in the seconds 
   since start of the file
'''
def load_scene(video_fname,start,end,verbose=False,crop_size=256):
    print(video_fname)
    cap = cv2.VideoCapture(video_fname)
    print('cap',cap.isOpened())
    cap.set(cv2.CAP_PROP_POS_MSEC,start)    
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    print('Frame timestamp:{}'.format(timestamp))
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()   
        if ret:        
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
            frame = cv2.resize(frame,(crop_size,crop_size), interpolation = cv2.INTER_CUBIC)
            frames.append(np.array(frame))
            if verbose:
                cv2.imshow('frame',frame)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)        
                if timestamp > end:
                    break            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)
        else:     
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frames
    
    
def scene_generator(caption_fname,video_folder):
    captions = utils.load_caption(caption_fname)
    availabel_videos = glob.glob(os.path.join(video_folder,'*.mp4'))
    availabel_videos_keys = [x.split('.')[0].split('/')[-1] for x in availabel_videos]
    for video_key in availabel_videos_keys[10:]:    
        if video_key in captions:
            scene_meta = captions[video_key]
            print('video_key:{},duration:{}'.format(video_key,scene_meta['duration']))
            for index in range(len(scene_meta['timestamps'])):
                timestamp = scene_meta['timestamps'][index]
                sentence = scene_meta['sentences'][index]
                print(timestamp,sentence)
                video_fname = os.path.join(video_folder,video_key+".mp4")
                frames = load_scene(video_fname=video_fname,
                    start=float(timestamp[0])*1000.,
                    end=float(timestamp[1])*1000.)
                
                yield frames,sentence           
        else:
            print('Skip video:{}'.format(video_key))
        
    
    
    
if __name__ == "__main__":
    config = {
            'caption_fname':utils.CAPION_METADATA_FILE,
            'video_folder' :utils.VIDEO_FOLDER
    }
    
    g = scene_generator(**config)
    frames,sentence = next(g)
    print(frames, sentence)
    
    
    