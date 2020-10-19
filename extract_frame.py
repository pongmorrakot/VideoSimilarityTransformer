#!/usr/bin/env python
# coding: utf-8

# ## 关键帧提取

# In[1]:
import subprocess
import math
import os
import sys
import glob
import shutil
import codecs
from tqdm import tqdm_notebook as tqdm

import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

#get_ipython().run_line_magic('pylab', 'inline')
from PIL import Image

# In[2]:


IN_PATH = './UCF-101'
OUT_PATH = './UCF-101-Imgs'


# In[34]:


# 抽取关键帧
class FrameExtractor():
    # key uniform scene
    def __init__(self, inpath, outpath):
        self.video_root_path = inpath 
        self.output_root_path = outpath  
        self.video_file_paths = self._get_videos(self.video_root_path)
        if not os.path.exists(self.output_root_path):
            os.mkdir(self.output_root_path)
        
    def _get_videos(self, path):
        video_file_paths = glob.glob(path + '/*/*.avi')
        #print(video_file_paths)
        return video_file_paths
    
    def extract_keyframe(self, video_path, frame_path):
        video_id = video_path.split('/')[-1][:-4]
        if not os.path.exists(frame_path + video_id):
            os.mkdir(frame_path + video_id)

        # 抽取关键帧（I帧）
        command = ['ffmpeg', '-i', video_path,
                   '-vf', '"select=eq(pict_type\,I)"',
                   '-vsync', 'vfr', '-qscale:v', '2',
                   '-f', 'image2',
                   frame_path + '{0}/{0}_%05d.jpg'.format(video_id)]
        os.system(' '.join(command))

        # 抽取视频关键帧时间
        command = ['ffprobe', '-i', video_path,
                   '-v', 'quiet', '-select_streams',
                   'v', '-show_entries', 'frame=pkt_pts_time,pict_type|grep',
                   '-B', '1', 'pict_type=I|grep pkt_pts_time', '>',
                   frame_path + '{0}/{0}.log'.format(video_id)]
        os.system(' '.join(command))
    
    def _extract_keyframe(self, param):
        self.extract_keyframe(param[0], param[1])

    def extract_uniformframe(self, video_path, frame_path, frame_per_sec=1):
        
        #print("video_path = {}".format(video_path))
        #print("frame_path = {}".format(frame_path))
        
        tmp = video_path.split('/')
        video_type = tmp[-2]
        video_id = tmp[-1] 
        

        #if not 'v_PlayingGuitar_g25_c05' in video_id: 
        #    return  

        #print("video_type = {}, video_id = {}".format(video_type,video_id)) 
      
        video_type_path = frame_path + '/'+ video_type + '/' 
        video_file_path = frame_path + '/'+ video_type + '/' + video_id +'/'
        
        if not os.path.exists(video_type_path):
            os.mkdir(video_type_path)
        
        if not os.path.exists(video_file_path):
            os.mkdir(video_file_path)
        else: 
            return 
                
        print("Processing video_path = {}".format(video_path))
        
        # -r 指定抽取的帧率，即从视频中每秒钟抽取图片的数量。1代表每秒抽取一帧。
        command = ['ffmpeg', '-i', video_path,
                   '-r', str(frame_per_sec),
                   '-q:v', '2', '-f', 'image2',
                   video_type_path + '{0}/{0}_%4d.jpg'.format(video_id)]
        os.system(' '.join(command))

    
    def _extract_uniformframe(self, param):
        self.extract_uniformframe(param[0], param[1], param[2])
    
    # 关键帧用时间戳重命名
    def _rename(self, video_paths, frame_path, mode='key', frame_per_sec=1):
        for path in video_paths[:]:
            video_id = path.split('/')[-1][:-4]
            id_files = glob.glob(frame_path + video_id + '/*.jpg')
            # IMPORTANT!!!
            id_files.sort()
            if mode == 'key':
                id_times = codecs.open(frame_path + '{0}/{0}.log'.format(video_id)).readlines()
                id_times = [x.strip().split('=')[1] for x in id_times]

                for id_file, id_time in zip(id_files, id_times):
                    shutil.move(id_file, id_file[:-9] + id_time.zfill(15) + '.jpg')
            else:
                id_time = 0.0
                for id_file in id_files:
                    shutil.move(id_file, id_file[:-19] + '{:0>15.4f}'.format(id_time) + '.jpg')
                    id_time += 1.0 / frame_per_sec

    def extract(self, mode='key', num_worker=5, frame_needed=64):
        if mode == 'key':
            pool = Pool(processes=num_worker)
            for path in self.train_query_paths:
                pool.apply_async(self._extract_keyframe, ((path, self.train_path + 'query_keyframe/'),))

            pool.close()
            pool.join()

        elif mode == 'uniform':
            '''           
            pool = Pool(processes=num_worker)
            for path in self.video_file_paths:
                pool.apply_async(self._extract_uniformframe, ((path, self.output_root_path, frame_per_sec_q),))

            pool.close()
            pool.join()
             
            '''
            for path in self.video_file_paths:
                frame_num = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                frame_num = int(frame_num.stdout)
                frame_per_sec = math.floor(frame_needed/frame_num)
                self.extract_uniformframe(path, self.output_root_path, frame_per_sec)
            
        else:
            None


# In[35]:


frame_extractor = FrameExtractor(IN_PATH, OUT_PATH)


# In[36]:
frame_extractor.extract(mode='uniform', num_worker=16, frame_needed=64)


# In[ ]:




