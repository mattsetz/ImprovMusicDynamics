####################################################
# Compute two indices of CSD:
# lagged autocorrelation + variability
#
# MS 8/21/19
####################################################

import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial.distance import pdist


IN_DIR = '../Librosa-Pipeline/mfcc/'
OUT_DIR = '../Librosa-Pipeline/csd/'
STEP_SIZE = 0.2 # seconds
cols = ['X'+str(i+1) for i in range(4)] # mfc coefficients to consider


##############################################
# FUNCTIONS
##############################################

def first_differencing(mfcc_df):
	for c in cols:
		mfcc_df[c] = mfcc_df[c].diff()
	return mfcc_df[['t']+cols]


def acf_helper(mfcc_t):
	acfs = []
	for c in cols:
		acfs.append(mfcc_t[c].autocorr())
	return np.mean(acfs)

def get_autocorrelation(mfcc_df, times, window):
	print("get_acf. window: "+str(window))
	acf = []
	for t in times:
		mfcc_t = mfcc_df.loc[(mfcc_df.t>(t-window))&(mfcc_df.t<=t)]
		acf.append(acf_helper(mfcc_t))
	return acf



def var_helper(mfcc_t):
	distances = pdist(mfcc_t[cols].values)
	return np.nanmean(distances)

def get_variability(mfcc_df, times, window):
	print("get_var")
	var = []
	for t in times:
		mfcc_t = mfcc_df.loc[(mfcc_df.t>(t-window))&(mfcc_df.t<=t)]
		var.append(var_helper(mfcc_t))
	return var


def compute_csd_indices(mfcc_filepath):
	print("compute_csd_indices")

	# read in mfcc data
	mfcc = pd.read_csv(mfcc_filepath)

	# pre-process time series
	mfcc = first_differencing(mfcc)

	# compute lag-1 acf + variability across range of window sizes
	csd = pd.DataFrame()
	windows = [2,5,10,20]
	for w in windows:
		times = np.arange(w,max(mfcc['t'])+STEP_SIZE,STEP_SIZE)
		acf = get_autocorrelation(mfcc, times, w)
		var = get_variability(mfcc, times, w)
		csd = pd.concat((csd, pd.DataFrame({'t': times, 'acf': acf,
											'var': var, 'window': w})))

	# write out dataframe
	outname = mfcc_filepath.split('/')[-1].replace('-mfcc','-csd')
	csd.to_csv(OUT_DIR+outname, index = False)
	return


##############################################
# MAIN
##############################################
infiles = glob(IN_DIR+"*")
for f in infiles:
	print(f)
	compute_csd_indices(f)
