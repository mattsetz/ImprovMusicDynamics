#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
from glob import glob
import pandas as pd
import numpy as np

WINDOW_SIZE = 2
STEP_SIZE = 0.1

in_dir = '../Librosa-Pipeline/finalFiles/'
out_dir = '../Librosa-Pipeline/mfcc/'


# In[2]:


def extract_mfcc(y, sr, name):
    # extract mfcc
    frame_len = int(sr*WINDOW_SIZE)
    hop_len = int(sr*STEP_SIZE)
    mfcc = librosa.feature.mfcc(y=y,sr=sr,
                                n_fft=frame_len,
                                hop_length=hop_len)
    
    # format data frame
    mfcc = pd.DataFrame(np.transpose(mfcc))
    mfcc.columns = ['X'+str(i+1) for i in range(len(mfcc.columns))]
    mfcc['t'] = WINDOW_SIZE + mfcc.index*STEP_SIZE
    mfcc['name'] = name
    mfcc.to_csv(out_dir+name+'-mfcc.csv',index=False)
    
    return


# In[8]:


in_files = glob(in_dir+'*')
for f in in_files:
    name = f.split('/')[-1].replace('.wav','').replace('.mp3','').replace('.aif','')
    print(name)
    
    try:
        y, sr = librosa.load(f)
    except:
        print('error loading ' + name)
        continue

    try:
        extract_mfcc(y, sr, name)
    except:
        print('error processing ' + name)

