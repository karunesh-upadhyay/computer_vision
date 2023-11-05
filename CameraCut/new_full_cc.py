
# coding: utf-8

# In[1]:

import time
import pandas as pd
import numpy as np
import cv2
import os
import glob
import multiprocessing as mp
from scipy.spatial.distance import cosine,euclidean
import pickle
import sys
from itertools import groupby
from operator import itemgetter

# In[2]:


# frame path

start_time  = time.time()
file_name = sys.argv[1]

output_path=sys.argv[2]

#file_name = glob.glob('/home/ao-collab/v2f/frames/*')[0].split('/')[-1]
frame_path = '/home/ao-collab/v2f/frames/' + file_name.split('.')[0] + '/'
frames = sorted(glob.glob(frame_path + '*.jpg' ), key = lambda x : int(x.split(frame_path)[1].split('.jpg')[0]))
print(file_name)##,file_name.split('.')[0])
frame_color_embeddings_path = '/home/ao-collab/camera_cut/frame_color_embeddings/' + file_name.split('.')[0] +'_'+str(16)+ '.npy'
final_shot_embeddings_path = '/home/ao-collab/camera_cut/shot_embeddings/' + file_name.split('.')[0] + '.csv'
color_cc_path = '/home/ao-collab/camera_cut/color_cameracuts/' + file_name.split('.')[0] + '.csv'

#final_camera_cut_after_black_path = '/home/ao-collab/camera_cut/final_color_cc/' + file_name + '.csv'
#print (frames[0:5])

black_embedding = np.load('/home/ao-collab/black_safe/black_embedding_4096.npy')

black_embedding =np.array([1]+[0]*255)
final_black_cc_path = '/home/ao-collab/camera_cut/final_color_cc/'


# In[ ]:


def euc_dist(x, y):
    """Calculate Euclidean distance from x to y
    """
    return np.sqrt(sum(np.square(x-y)))


# In[ ]:


# hist

def histogram_gen(image_path):
    #print (image_path)
    image = cv2.imread(image_path)
    #images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    #hist = cv2.calcHist([a], [0], None, [32], [0,256])
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hist = cv2.calcHist([hsv], [0,1,2], None, [16,8,8], [0, 180,0,256,0,256])
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update

    hist = cv2.calcHist([image], [0, 1, 2], None, [16,16,16],[0, 256, 0, 256, 0, 256])
    #hist = cv2.calcHist([image], [0, 1, 2], None, [4,4,4],[0, 256, 0, 256, 0, 256])
    
    #hist = hist.flatten().reshape(len(hist),-1)
    #hist = hist/(max(hist))
    hist = cv2.normalize(hist, hist).flatten()
    hist = hist.flatten()
    #hist = scaler.fit_transform(hist)
    return hist


# In[ ]:


#black = histogram_gen(frames[720])
#np.save('black_embedding_4096.npy',black)


# In[ ]:


'''
pool = mp.Pool()
frame_color = pool.map(histogram_gen,frames)
frame_color_npy = np.array(frame_color)

#np.save('check_p.npy',frame_color_npy)
np.save(frame_color_embeddings_path,frame_color_npy)
'''


# In[ ]:


# camera cuts
color_hash = np.load(frame_color_embeddings_path)
#color_hash = np.load('check_p.npy')


# In[ ]:


# identify perfect sequences
reduction_number = 1
image_correlation_threshold = 0.8
#image_correlation_threshold = 0.95

distance_csv_color = [cv2.compareHist(np.array(color_hash[i]),np.array(color_hash[i+reduction_number]),method = cv2.HISTCMP_CORREL) for  i in range(0, len(color_hash)-reduction_number)]
#distance_csv_color = [1-cosine(color_hash[i],color_hash[i+reduction_number]) for  i in range(0, len(color_hash)-reduction_number)]

distance_csv = pd.DataFrame([distance_csv_color]).transpose()
distance_csv.columns = [ 'image_color_cor']
distance_csv['frame_i'] = range(0,len(distance_csv))
distance_csv['frame_j'] = distance_csv['frame_i'].apply(lambda x: x+reduction_number)
distance_csv.head(10)


# In[ ]:


# filter 1  color filter less than color threshold
#print (image_correlation_threshold)
#cut_thresh  = distance_csv[distance_csv['image_color_cor']<= image_correlation_threshold].copy()
cut_thresh  = distance_csv[distance_csv['image_color_cor']<= image_correlation_threshold].copy()

cut_thresh.reset_index(drop=True, inplace=True)
start_filter_1 = cut_thresh.frame_j.tolist()
start_filter_1.insert(0,0)


# In[ ]:


num_frames = len(distance_csv)
cc_frame_start = start_filter_1
cc_start_copy = cc_frame_start
cc_frame_start_old=cc_frame_start[:]
break_key=0
start_key=0

while ((break_key==0) and (start_key==0 or cc_frame_start_old!=cc_frame_start)):
    cc_frame_start_old=cc_frame_start[:]
    start_key=1
#         print("WHILE_CHECK")
    for i in range(1,len(cc_frame_start)-1):
#             print("I",i)
        if (cc_frame_start[i+1]-cc_frame_start[i])<7:
            #print("I",i,cc_frame_start[i])
#                 print(get_hue_values([cc_frame_start[i]]))
            diff1=euc_dist(color_hash[cc_frame_start[i]] , color_hash[cc_frame_start[i-1]]) 
            diff2=euc_dist(color_hash[cc_frame_start[i]] , color_hash[cc_frame_start[i+1]]) 
            
            
            
            #print(diff1,diff2)
            if diff1<diff2 or diff1==diff2:
                del cc_frame_start[i]
#                     print("BREAK")
                break
            elif diff1>diff2:
                del cc_frame_start[i+1]
#                     print("BREAK")
                break
    if num_frames not in cc_frame_start:
        cc_frame_start.append(num_frames)
#         print("W1") 
    break_key=1
    for i in range(1,len(cc_frame_start)-1):
        if (cc_frame_start[i+1]-cc_frame_start[i])<7:
            break_key=0
#         print("W2",cc_frame_start_old==cc_frame_start)
#         print(break_key==0,start_key==0 ,cc_frame_start_old!=cc_frame_start)

start=[cc_frame_start[0]]
end=[]
#print(cc_frame_start[0:5])
for i in range(1,len(cc_frame_start)-1):
#         print(i,cc_frame_start[i+1],cc_frame_start[i])
    if cc_frame_start[i+1]-cc_frame_start[i] > 7:        
        start.append(cc_frame_start[i])
if start[1]-start[0]<7:
    del start[1]

for i in range(1,len(start)):
    end.append(start[i]-1)

#     start.append(cc_frame_start[i])
end.append(cc_frame_start[-1])

df=pd.DataFrame({'start_frame':start ,'end_frame':end })
df=df[['start_frame','end_frame']]
#for i in range(len(start)):
    #print(start[i],end[i],end[i]-start[i])

df_new =df.copy()
#df.to_csv(mapper_csv_path,index=False)


# In[ ]:


#print('# POST MISSILE')
df = df_new
total_shots=len(df)
mast_s_no=range(1,total_shots+1)
mast_s_start=list(df['start_frame'])
mast_s_end=list(df['end_frame'])



def avg_frame_embs_master(num):
#     print(num)
    try:
        
        temp1=color_hash[num-1]
        temp2=color_hash[num-2]
        temp3=color_hash[num-3]
        temp4=color_hash[num-4]
        
        
        
        
        avg_first=(temp1+temp2+temp3+temp4)/4.
        #avg_first=(temp1+temp2+temp3+temp4)/4.

    
        temp5=color_hash[num+1]
        temp6=color_hash[num+2]
        temp7=color_hash[num+3]
        temp8=color_hash[num+4]
       
       
       
        #avg_second=(temp4+temp5)/2.
        avg_second=(temp5+temp6+temp7+temp8)/4.

    except:
        avg_second=avg_first

    #return euc_dist(avg_first,avg_second)
    
    return cv2.compareHist(avg_first,avg_second,method = cv2.HISTCMP_CORREL)
    #return 1-cosine(avg_first,avg_second)


# In[ ]:


# solving case of one frame kios


temp_master=[]


for i in range(0,total_shots-1):
#     print("#######")
    lambdaa=avg_frame_embs_master(mast_s_end[i])
#     print(lambdaa)
    temp_master.append(lambdaa)
#         temp.append(i+1)
    #print(i+1,lambdaa,mast_s_start[i],mast_s_end[i])


# In[ ]:


threshold2 = 0.8
temp=[]
for i in range(len(temp_master)):
    #print( temp_master[i])
    if temp_master[i]>threshold2:
            #print(i+1,temp_master[i],mast_s_start[i],mast_s_end[i]+1)
            temp.append(i+1)


# In[ ]:


mu=np.average(temp_master)
sigma=np.std(temp_master)
#print('mu,sigma',mu,sigma)
#print('temp',len(temp),temp)


# In[ ]:


def start_end_set(L):   
        start=[L[0]]
        end=[]
        if L[1]-L[0]!=1:
            end.append(L[0])
            start.append(L[1])
        for i in range(1,len(L)-1):
            if L[i+1]-L[i]!=1:
                end.append(L[i])
                start.append(L[i+1])
        end.append(L[-1])
        return (start,end)


# In[ ]:


try:
    
    if len(temp)>0:
        start_missile,end_missile=start_end_set(temp)

        #print ('hi')
        #print(start_missile,end_missile)

        missile_solved_cc_master=[]

        # joining the next shot on missile shots
        for i in range(len(start_missile)):
            #print (i,start_missile[i])
            if end_missile[i]!=mast_s_no[-1]:
                missile_solved_cc_master.append(sorted(list(range(start_missile[i],end_missile[i]+1))+[end_missile[i]+1]))
            else:
                missile_solved_cc_master.append(sorted(range(start_missile[i],end_missile[i]+1)))

        # getting all the missile shots
        temp=[j for i in missile_solved_cc_master for j in i]
        # print(temp)

        new_cc_shots=[]
        for i in mast_s_no:
            if i not in temp:
                new_cc_shots.append([i])
        new_cc_shots+=missile_solved_cc_master
        # print("#######")
        # print(new_cc_shots)
        # print("#######")
        temp=[i[0] for i in new_cc_shots]
        df=pd.DataFrame({'mast': new_cc_shots ,'temp':temp})
        df=df.sort_values('temp')
        # print(list(df['mast']),len(list(df['mast'])))
        # print("#######")
        new_cc_shots=list(df['mast'])
        # print(new_cc_shots)


        missile_cc_start=[]
        missile_cc_end=[]
        for i in new_cc_shots:
            missile_cc_start.append(mast_s_start[min(i)-1])
            missile_cc_end.append(mast_s_end[max(i)-1])
        # print(missile_cc_start,len(missile_cc_start))
        # print("#####")
        # print(missile_cc_end,len(missile_cc_end))

    else:
        missile_cc_start=mast_s_start
        missile_cc_end=mast_s_end

    #for i in range(len(missile_cc_end)):
        #print(i+1,missile_cc_start[i],missile_cc_end[i])
        
        
except:
    
    missile_cc_start = df_new['start_frame'].tolist()
    missile_cc_end = df_new['end_frame'].tolist()


# In[ ]:


#print("CORRECT 1 frame misalignment")
load_ind=[]

missile_cc_start_copy=missile_cc_start[:]

for i in range(1,len(missile_cc_start)-1):
    should_merge=''
    #print(i+1,missile_cc_start[i],missile_cc_end[i])
    minus_3=missile_cc_start[i]-3
    minus_4=missile_cc_start[i]-4
    plus_3=missile_cc_start[i]+2
    plus_4=missile_cc_start[i]+3
    image1=missile_cc_start[i]-2
    image2=missile_cc_start[i]-1
    image3=missile_cc_start[i]
    image4=missile_cc_start[i]+1

#     print('minus_3',minus_3)
#     print('minus_4',minus_4)
#     print('plus_3',plus_3)
#     print('plus_4',plus_4)
#     print('image1',image1)
#     print('image2',image2)
#     print('image3',image3)
#     print('image4',image4)

    image1=color_hash[image1]        #-2
    image2=color_hash[image2]        #-1
    image3=color_hash[image3]        #original
    image4=color_hash[image4]        #+1   
    minus_3=color_hash[minus_3]       #-3
    minus_4=color_hash[minus_4]       #-4 
    plus_3=color_hash[plus_3]        #+2
    plus_4=color_hash[plus_4]        #+3

#     print("###",image1)
    image1_A=(euc_dist(image1,(minus_4+minus_3)/2.))    
    image1_B=(euc_dist(image1,(plus_3+plus_4)/2.))    

#     print("###",image2)
    image2_A=(euc_dist(image2,(minus_4+minus_3)/2.)) 
    image2_B=(euc_dist(image2,(plus_3+plus_4)/2.)) 

#     print("###",image3)
    image3_A=(euc_dist(image3,(minus_4+minus_3)/2.)) 
    image3_B=(euc_dist(image3,(plus_3+plus_4)/2.)) 

#     print("###",image4)
    image4_A=(euc_dist(image4,(minus_4+minus_3)/2.)) 
    image4_B=(euc_dist(image4,(plus_3+plus_4)/2.)) 


    #####
    if image1_A<image1_B:
        should_merge+='b'
    elif image1_A>image1_B:
        should_merge+='f'

    if image2_A<image2_B:
        should_merge+='b'
    elif image2_A>image2_B:
        should_merge+='f'

    if image3_A<image3_B:
        should_merge+='b'
    elif image3_A>image3_B:
        should_merge+='f'

    if image4_A<image4_B:
        should_merge+='b'
    elif image4_A>image4_B:
        should_merge+='f'       

    #print(should_merge)
#     print('12',euc_dist(image1,image2))
#     print('23',euc_dist(image2,image3))
#     print('34',euc_dist(image3,image4))

#     if should_merge == 'bfff': # or should_merge == 'bbbf':
#         print('image1_A',image1_A,image1_B)
#         print('image2_A',image2_A,image2_B)
#         print('image3_A',image3_A,image3_B)
#         print('image4_A',image4_A,image4_B)
#         print(missile_cc_start[i])
#         print(should_merge)

#     print(should_merge!='bbff')

    if should_merge!='bbbf' and should_merge!='bbff' and should_merge!='bfff':
        #print('yes')
        load_ind.append(missile_cc_start[i])

    elif should_merge=='bbbf':
        missile_cc_start_copy[i]+=1

    elif should_merge=='bfff':
        missile_cc_start_copy[i]-=1

print(load_ind)

for i in range(len(load_ind)):
    del missile_cc_start_copy[np.where(np.array(missile_cc_start_copy)==load_ind[i])[0][0]]

missile_cc_end_copy=[]
missile_cc_start_copy = list(missile_cc_start_copy)
for i in range(1,len(missile_cc_start_copy)):
    missile_cc_end_copy.append(missile_cc_start_copy[i]-1)
missile_cc_end_copy.append(max(missile_cc_end))

missile_cc_start = missile_cc_start_copy
missile_cc_end = missile_cc_end_copy


# In[ ]:


cc_final = pd.DataFrame([missile_cc_start,missile_cc_end]).T
cc_final.columns = ['Start_Frame', 'End_Frame']
cc_final['s_no'] = range(1,len(cc_final)+1)
cc_final.to_csv(final_black_cc_path + file_name.split('.')[0] + '.csv',index =False) 
name_file= file_name.split('.')[0] + '.csv'
cmd=""" gsutil -m cp -r  {final_black_cc_path}{name_file} {output_path}/ """.format(final_black_cc_path=final_black_cc_path,name_file=name_file,output_path=output_path)
#os.system(cmd)

print("Embbedding csv generated")

# In[ ]:


print ('post missile done')


# In[ ]:


# black check

#black_detected = [i for i in range(len(color_hash)) if (color_hash[i] == black_embedding).all()]
black_detected = [i for i in range(len(color_hash)) if cosine(color_hash[i],black_embedding)<=0.0002]

len(black_detected)

# consecuitve sequence detector
black_shot_combined = list()

for k, g in groupby(enumerate(black_detected), lambda ix : ix[0] - ix[1]):
    black_shot_combined.append(list(map(itemgetter(1), g)))
black_shot_combined


# In[ ]:


color_start = cc_final['Start_Frame'].tolist()
black_start= [x[0] for x in black_shot_combined if len(x)>=7]
black_start
black_next_start= [x[-1]+1 for x in black_shot_combined if len(x)>=7]
black_next_start = [x for x in black_next_start if x<=len(distance_csv)]
black_next_start


# In[ ]:


# shot removal near black
# new_cc

new_cc_start= color_start + black_start + black_next_start
new_cc_start = np.unique(new_cc_start)
#print (new_cc_start)

drop_list= list()
for i in range(len(new_cc_start)):
    # black_start_check
    current = new_cc_start[i]
    #print (current)
    if current == 0: next

    elif current == new_cc_start[-1] : next    
    elif current == len(distance_csv): 
        drop_list.append(i)

    
    elif current in black_start:
        
        if abs(current - new_cc_start[i-1]) <=10:
            drop_list.append(i-1)
        else : next
            
    elif current in black_next_start:
        if  abs(current - new_cc_start[i+1]) <=10:
            drop_list.append(i+1)
        else : next
            
    else : next
            
dropped = [new_cc_start[i] for i in drop_list]
#print (dropped)
new_cc_start = np.delete(new_cc_start,drop_list).tolist()


if new_cc_start[0]!=0:
    new_cc_start.insert(0,0)


# In[ ]:


new_cc_end = [new_cc_start[x+1]-1 for x in range(len(new_cc_start)-1)]
new_cc_end.append(len(distance_csv))
final_black_cc = pd.DataFrame([new_cc_start,new_cc_end]).T

final_black_cc.columns = ['Start_Frame','End_Frame']
# final sevenshot check
new_cc_start2 = final_black_cc['Start_Frame'].values
#print (type(new_cc_start2))
seven_check = np.where(final_black_cc['End_Frame'] - final_black_cc['Start_Frame']+1 < 7)[0]
new_cc_start2 =  np.delete(new_cc_start2,seven_check).tolist()
if new_cc_start2[0]!=0:
    new_cc_start2.insert(0,0)


# In[ ]:


new_cc_end2 = [new_cc_start2[x+1]-1 for x in range(len(new_cc_start2)-1)]
new_cc_end2.append(len(distance_csv))
final_black_cc2 = pd.DataFrame([new_cc_start2,new_cc_end2]).T

final_black_cc2.columns = ['Start_Frame','End_Frame']


final_black_cc2['s_no'] = range(1,len(final_black_cc2)+1)

final_black_cc2.to_csv(final_black_cc_path + file_name.split('.')[0] + '.csv',index =False)

name_file=file_name.split('.')[0] + '.csv'
cmd= """ gsutil -m cp -r  {final_black_cc_path} {output_path} """.format(final_black_cc_path=final_black_cc_path,name_file=name_file,output_path=output_path)

os.system(cmd)
print('GSUTIL',cmd)
print("final black path csv generated")
# In[ ]:


# black info 
#black_shot= [ final_black_cc2.loc[i,'s_no'] for i in range(len(final_black_cc2)) if final_black_cc2.loc[i,'Start_Frame'] in black_start]

#black_shot = pd.DataFrame(black_shot)
#black_shot.columns= ['shot_no']
#black_shot.to_csv(final_black_cc_path + 'black_info_' + file_name +'.csv',index = False)

end_time = time.time()
print ('####################################################')
print ('camera cut generation done succesfully')
print ('camera cut time taken:{x} secs'.format(x=end_time-start_time))
