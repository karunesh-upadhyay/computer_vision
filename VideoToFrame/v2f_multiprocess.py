import sys
import ffmpeg
import subprocess
import os
from moviepy.editor import VideoFileClip
import pandas as pd
import cv2
import itertools
import time
import numpy as np
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
import glob
import multiprocessing
import time
import json

v2f_start= time.time()

episodes=[str(sys.argv[1])]
episode_name=[]
new_mov_time_=[]
old_mov_time_=[]
cv2_time_=[]

def get_frame_rate(filename):
    """DocString
    """
    if not os.path.exists(filename):
        print('Files not exists')
        return -1
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    print(out)
    out = str(out)
    rate = out.split('=')[1].strip()[1:-1].split('/')
    print("rate : ", rate)
    if len(rate) == 1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1].split('"')[0])
    return -1
    
    
def get_frames1(iterr):
    k_t=time.time()
    ffmpeg_videoreader = FFMPEG_VideoReader(filename=video_path)
    
    start=min(iterr*2000,derived_total_frames)
    end=min(derived_total_frames,(iterr+1)*2000)
    count=start
    for i in range(start,end):
#             if i%2000==0:
#                 print('I',i)
        frame = ffmpeg_videoreader.get_frame(i*(float(1/frame_rate)))
        #cv2.imwrite('/home/aodev/v2f/LBC503339/LBC503339_{c}.jpg'.format(c=count), cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),(640,360)))
        cv2.imwrite('/home/ao-collab/v2f/frames/' + episodes[epi_name] + '/{c}.jpg'.format(c=count),cv2.cvtColor(frame, cv2. COLOR_RGB2BGR))
        count+=1
    print(iterr,time.time()-k_t)
    return ()
    


    
    
#for epi_name in range(len(episodes))[0:1]:
for epi_name in range(len(episodes)):
    print(epi_name)
    #video_path=glob.glob('/home/aodev/v2f/'+episodes[epi_name]+'.*')[0]
    video_path=glob.glob('/home/ao-collab/v2f/video/'+episodes[epi_name]+'.*')[0]
    print(video_path)


    # metadata info

    video_info = ffmpeg.probe(video_path)
    video_json = json.dumps(video_info)
    with open('/home/ao-collab/' + episodes[epi_name] + '/metadata.json', 'w') as fp:
        json.dump(video_json, fp)




    episode_name.append(episodes[epi_name])
    print(episode_name[-1])
    if not os.path.exists('/home/ao-collab/v2f/frames/' + episodes[epi_name]):
        os.makedirs('/home/ao-collab/v2f/frames/' + episodes[epi_name])

    frame_rate = get_frame_rate(video_path)
    print('frame_rate',frame_rate)
    ffprobe_time = float(ffmpeg.probe(video_path)['streams'][0]['duration'])
    print('ffprobe_time',ffprobe_time)
    derived_total_frames = round(ffprobe_time * frame_rate)
    
    print(derived_total_frames, ffprobe_time)

    ################################################################
    ################################################################
    ################################################################

    start_time=time.time()
#     ffmpeg_videoreader = FFMPEG_VideoReader(filename=video_path)

    start_time=time.time()

    print('Total_iterations:',list(range(int(np.ceil(derived_total_frames/2000)))))
    
    Total_iterations=int(np.ceil(derived_total_frames/2000))
    
    pool=multiprocessing.Pool()
    k=pool.map(get_frames1,list(range(Total_iterations)))
    
#     for iter_num in range(Total_iterations):
#         get_frames1(iter_num)
        
    pool.close()

    pool.join()

    print(time.time()-start_time)

    new_code_time=time.time()-start_time
    
    new_mov_time_.append(new_code_time)



v2f_end = time.time()
new_code_time= v2f_end - v2f_start
print ('####################################################')
print ('v2f done succesfully')
print ('v2f time taken:{x} secs'.format(x=new_code_time))
