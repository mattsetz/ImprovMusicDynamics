#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from glob import glob


# In[9]:


MATRIX_STEP = .5 # interval btn successive entries in dissimilarity matrix

in_dir = '../Librosa-Pipeline/dissimilarity-matrix-500ms/'
out_dir = '../Librosa-Pipeline/stability/'


# In[10]:


##############################
# Compute instability
##############################

def instability(diss_m, window_size):
    instable = [None]*len(diss_m)
    for i in np.arange((window_size), len(diss_m), dtype=int):        
        beg = int(i-(window_size))
        end = i
        instable[i] = np.mean(diss_m[beg:end,beg:end])
    
    return instable



def get_stability(f):
    name = f.split('/')[-1].replace('-diss-matrix.csv','')
        
    diss_m = pd.read_csv(f).values
        
    # compute stability
    instable10 = instability(diss_m, 20)
    instable30 = instability(diss_m, 60)
    instable60 = instability(diss_m, 120)

    # format data frame
    instable = pd.DataFrame({'t': [MATRIX_STEP*(i+1) for i in range(len(diss_m))],
                             'instable10': instable10,
                             'instable30': instable30,
                             'instable60': instable60})
    instable.to_csv(out_dir+name+'-instability.csv',index=False)

    return

##############################
# Read in files
##############################
print("freejazz_stability")
in_files = glob(in_dir+'*')
for f in in_files:
    print(f)
    try:
        get_stability(f)
    except:
        print("error processing " + f)