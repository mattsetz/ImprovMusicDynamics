#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics.pairwise import euclidean_distances



def compute_dissimilarity_500ms(mfcc, name, out_dir_matrix):
    # compute dissimilarity matrix
    mfcc_cols = [col for col in mfcc if col.startswith('X')]
    mfcc_vals = mfcc[mfcc_cols].iloc[::5].values
    diss = euclidean_distances(mfcc_vals)
    diss_df = pd.DataFrame(diss).to_csv(out_dir_matrix+name+'-diss-matrix.csv',index=False)
    
    return


def get_similarity_500ms():
    print("get_similarity_500ms")

    in_dir = '../Librosa-Pipeline/mfcc/'
    out_dir = '../Librosa-Pipeline/dissimilarity-500ms/'
    out_dir_matrix = '../Librosa-Pipeline/dissimilarity-matrix-500ms/'

    in_files = glob(in_dir+'*')

    for f in in_files:
        name = f.split('/')[-1].replace('-mfcc.csv','')
        print(name)

        mfcc = pd.read_csv(f)
        try:
            compute_dissimilarity_500ms(mfcc, name, out_dir_matrix)
        except:
            print('failed computing similarity: '+ name)
            continue
    
    return

get_similarity_500ms()
