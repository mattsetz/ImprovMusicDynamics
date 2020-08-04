#!/usr/bin/env python
# coding: utf-8

# In[4]:


import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


# In[1]:


in_dir = '../Librosa-Pipeline/dissimilarity-matrix-500ms/'
out_dir = '../Librosa-Pipeline/drift/'

MATRIX_STEP = .5 # 2 sec interval between successive enties in input dissimilarity matrix
WINDOW_SIZE_SEC = 30 # seconds
WINDOW_SIZE = int(WINDOW_SIZE_SEC/MATRIX_STEP)


# In[2]:


#########################
# Compute directed drift
#########################


def compute_drift_helper(m, name, WINDOW_SIZE):
  tau_vals = [None]*len(m)
  p_vals = [None]*len(m)
    
  for i in range(WINDOW_SIZE, len(m)-1):
    diss_i = m[(i-WINDOW_SIZE):(i-1),i]
    tau_i, p_i = -stats.kendalltau(diss_i, range(len(diss_i)))
    tau_vals[i] = tau_i
    p_vals[i] = p_i
    
  return (tau_vals, p_vals)


def compute_drift(m, name):
  taus10, p10 = compute_drift_helper(m, name, int(10/MATRIX_STEP))
  taus30, p30 = compute_drift_helper(m, name, int(30/MATRIX_STEP))
  taus60, p60 = compute_drift_helper(m, name, int(60/MATRIX_STEP))
    
  drift = pd.DataFrame({'t': [(i+1)*MATRIX_STEP for i in range(len(m))],
                        'tau10': taus10,
                        'p10': p10,
                        'tau30': taus30,
                        'p30': p30,
                        'tau60': taus60,
                        'p60': p60})
  drift.to_csv(out_dir+name+'-drift.csv',index=False)

  return


# In[5]:


def get_drift():
  print("get drift")
  infiles = glob(in_dir+'*')

  for f in infiles:
    name = f.split('/')[-1].replace('-diss-matrix.csv','')
    print(name)

    m = pd.read_csv(f).values
    compute_drift(m, name)
#try:
 #     compute_drift(m, name)
 #   except:
  #    print("error processing "+f)
    
  return

get_drift()
