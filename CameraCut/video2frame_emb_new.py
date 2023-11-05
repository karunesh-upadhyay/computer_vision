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
import os

v2f_start= time.time()
# episodes=['XRU007A.mov']
# episodes=['ABC192961.mp4']
episodes=[sys.argv[1]]
episode_name=[]
new_mov_time_=[]
old_mov_time_=[]
cv2_time_=[]

output_path=sys.argv[2]


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
    #print("rate : ", rate)
    if len(rate) == 1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1].split('"')[0])
    return -1
    
'''
def histogram_gen_new(bin_size,frame):
#     image = cv2.imread(image_path)
    hist = cv2.calcHist([frame], [0, 1, 2], None, [bin_size,bin_size,bin_size],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    hist = hist.flatten()
    return hist

'''


def histogram_gen_new(bin_size,frame):
    #read image

    # image = cv2.imread(image_path)
    
    image = np.array(frame)

    # number of sections an image is divided into (row-wise)
    num_image_sections=4
    
    # embeddings of each section of the image
    temp={}
    
    for num in range(num_image_sections):
        
        # cropping section of image
        image_cropped = cv2.cvtColor(image[int(num*image.shape[0]/num_image_sections):int((num+1)*image.shape[0]/num_image_sections)], cv2.COLOR_BGR2HSV)
        
        # get histograms
        hist = cv2.calcHist([image_cropped], [0], None, [bin_size],[0, 256])
        
        # normalise the embeddings and flatten
        hist = cv2.normalize(hist, hist).flatten()
        
        # save the embedding in the dictionary
        temp[num]=hist
    
    final_emb=list(temp[0])
    
    for num in range(1,num_image_sections):
        final_emb.extend(temp[num])

    return np.array(final_emb).flatten()





def get_frames1(iterr):
    print(iterr)
    k_t=time.time()
    ffmpeg_videoreader = FFMPEG_VideoReader(filename=video_path)
    
    start=min(iterr*2000,derived_total_frames)
    end=min(derived_total_frames,(iterr+1)*2000)
    count=start
    
    temp_em_4=[]
#     temp_em_16=[]
    for i in range(start,end):
        #if i%500==0:
            #print('I',i)
        frame = ffmpeg_videoreader.get_frame(i*(float(1/frame_rate)))
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),(320,180))
#         frame = cv2.resize(frame, (180,360))
#         print('frame',frame.shape)
#         cv2.imwrite('/home/aodev/v2f/LBC503339/LBC503339_{c}.jpg'.format(c=count), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#         cv2.imwrite('/home/aodev/audio_sim_ku/temp_imgs/frames/{c}.jpg'.format(c=count),cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),(320,180)))
        temp_em_4.append(histogram_gen_new(64,frame))
#         temp_em_16.append(histogram_gen_new(16,frame))
        count+=1
    #print(iterr,time.time()-k_t)
    return (temp_em_4)#,temp_em_16)


for epi_name in range(len(episodes)):
    #print(epi_name)
    #video_path=glob.glob('/home/aodev/v2f/'+episodes[epi_name]+'.*')[0]
    video_path=glob.glob('/home/ao-collab/input/'+episodes[epi_name])[0]
    #print(video_path)

    video_info = ffmpeg.probe(video_path)
    video_json = json.dumps(video_info)

    
    with open('/home/ao-collab/' + episodes[epi_name].split('.')[0] + '/metadata.json', 'w') as fp:
        json.dump(video_json, fp)
    
    episode_name.append(episodes[epi_name].split('.')[0])

    frame_rate = get_frame_rate(video_path)
    #print('frame_rate',frame_rate)
    ffprobe_time = float(ffmpeg.probe(video_path)['streams'][0]['duration'])
    #print('ffprobe_time',ffprobe_time)
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

    #print(time.time()-start_time)

    new_code_time=time.time()-start_time
    
    new_mov_time_.append(new_code_time)

k=np.array(k)

EMBED_4=[]

EMBED_16=[]

for i in range(len(k)):
    #print(np.array(k[i][0]).shape,np.array(k[i][1]).shape)
    EMBED_4.extend(k[i])
#     EMBED_16.extend(k[i][1])
    
EMBED_4=np.array(EMBED_4)
# EMBED_16=np.array(EMBED_16)
EMBED_4.shape

#print(EMBED_4.shape,EMBED_16.shape)
v2f_end = time.time()
new_code_time= v2f_end - v2f_start


frame_color_embeddings_path  = '/home/ao-collab/camera_cut/frame_color_embeddings/'

np.save(frame_color_embeddings_path + str(episodes[0].split('.')[0]) + '_4.npy',EMBED_4)
np.save(frame_color_embeddings_path + str(episodes[0].split('.')[0]) + '_16.npy',EMBED_4)


cmd = """ gsutil -m cp -r /home/ao-collab/camera_cut/frame_color_embeddings {output_path} """.format(output_path=output_path)
os.system(cmd)
print('USING GSUTIL', cmd)
#print ('####################################################')
print ('v2f done succesfully')
print ('v2f time taken:{x} secs'.format(x=new_code_time))
