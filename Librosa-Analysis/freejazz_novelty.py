#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from glob import glob
import re

in_dir = '../Librosa-Pipeline/dissimilarity-matrix-500ms/'
out_dir = '../Librosa-Pipeline/novelty/'


# In[3]:


# ASSUMES TIME INCREMENT OF 2 SECS

# LAG = time steps
def compute_novelty_by_lag(m,LAG):
    novelty = [None]*len(m)
    for i in range(30,len(m)-LAG):
        for j in range(-LAG,LAG):
            for k in range(-LAG,LAG):
                if (j*k >= 0): c = -1
                else: c = 1
                if (novelty[i] == None): novelty[i] = c*m[i+j,i+k]
                else: novelty[i] += c*m[i+j,i+k]
                
    return novelty

def compute_novelty(m, name):
    # compute novelty
    novel6 = compute_novelty_by_lag(m, 3)
    novel14 = compute_novelty_by_lag(m, 7)
    novel30 = compute_novelty_by_lag(m, 15)
    novel60 = compute_novelty_by_lag(m, 30)
    
    # format data frame
    novel_df = pd.DataFrame({'t': [2*(i+1) for i in range(len(m))],
                             'novelty6': novel6,
                             'novelty14': novel14,
                             'novelty30': novel30,
                             'novelty60': novel60})
    novel_df['name'] = name
    novel_df.to_csv(out_dir+name+'-novelty.csv',index=False)
    
    return novel_df


# In[4]:

def get_novelty():
    print("get_novelty")
    in_files = glob(in_dir+'*')
    # ToDo don't repeat processing

    for f in in_files:
        name = f.split('/')[-1].replace('-diss-matrix.csv','')
        print(name)
        
        diss = pd.read_csv(f)
        compute_novelty(diss.values[::4,::4], name) # select every 4th element to get 2 second bins

    return


# In[ ]:
get_novelty()


