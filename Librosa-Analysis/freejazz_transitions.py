####################################################
# Compute Kendall's tau for CSD indices
#
# MS 8/22/19
####################################################

import numpy as np
import pandas as pd
from glob import glob


IN_DIR = '../Librosa-Pipeline/novelty/'
OUT_DIR = '../Librosa-Pipeline/transitions/'
SMOOTH_WINDOW = 3 # time steps
TOO_CLOSE = 4 # seconds

##############################################
# FUNCTIONS
##############################################

def smooth_novelty(n):
	print("smooth_novelty")
	smooth_novel60 = []
	smooth_novel30 = []
	for i in range(len(n)):
		if (i<SMOOTH_WINDOW):
			smooth_novel60.append(None)
			smooth_novel30.append(None)
			continue
		smooth_novel60.append(np.mean(n.iloc[(i-3):(i+1)].novelty60))
		smooth_novel30.append(np.mean(n.iloc[(i-3):(i+1)].novelty30))
	n['smooth_novelty60'] = smooth_novel60
	n['smooth_novelty30'] = smooth_novel30
	return

def find_peaks(n,noveltyColumnName):
	print("find_peaks")
	time = []
	novelty = []
	
	is_increasing = False
	prev_novel = n.iloc[0][noveltyColumnName]
	for i in range(len(n)):
		cur_novel = n.iloc[i][noveltyColumnName]
		if (is_increasing and (cur_novel<prev_novel)):
			# found transition
			is_increasing = False
			time.append(n.iloc[i-1].t)
			novelty.append(prev_novel)

		if (cur_novel>=prev_novel):
			is_increasing = True

		prev_novel = cur_novel
	
	transitions = pd.DataFrame({'t': time,'novelty': novelty})
	transitions['normalized_novelty'] = transitions['novelty']/np.max(n[noveltyColumnName])
	mean_novelty = np.mean(n[noveltyColumnName])
	sd_novelty = np.std(n[noveltyColumnName])
	transitions['sdevs_from_mean'] = (transitions['novelty']-mean_novelty)/sd_novelty
	return transitions

# assumes time-sorted dataframe
def filter_peak_clusters(transitions_df):
	print("filter_peak_clusters")
	if len(transitions_df)==0:
		return transitions_df
	# identify rows to remove
	to_remove = []
	prev_row = transitions_df.iloc[0]
	for i in range(1,len(transitions_df)):
		cur_row = transitions_df.iloc[i]
		if (abs(prev_row.t-cur_row.t) < TOO_CLOSE):
			to_remove.append(i)
		else:
			prev_row = cur_row

	# remove those rows
	transitions_df = transitions_df.drop(to_remove)
	return transitions_df


def get_transitions(novelty_filepath):
	print("get_transitions")

	# Step 1. read in novelty time series
	novelty = pd.read_csv(novelty_filepath)

	# Step 2. smooth novelty time series
	smooth_novelty(novelty)

	# Step 3. find peaks in novelty
	transitions30 = find_peaks(novelty,'smooth_novelty30')
	transitions60 = find_peaks(novelty,'smooth_novelty60')

	# Step 4. filter peaks > novelty threshold. 2 stdevs from mean novelty
	'''
	mean_novelty30 = np.mean(novelty['smooth_novelty30'])
	sd_novelty30 = np.std(novelty['smooth_novelty30'])
	transitions30 = transitions30.loc[transitions30['novelty']>=(mean_novelty30+2*sd_novelty30)]

	mean_novelty60 = np.mean(novelty['smooth_novelty60'])
	sd_novelty60 = np.std(novelty['smooth_novelty60'])
	transitions60 = transitions60.loc[transitions60['novelty']>=(mean_novelty60+2*sd_novelty60)]
	'''

	# Step 5. select 1st peak in cluster of peaks
	transitions30 = filter_peak_clusters(transitions30)
	transitions60 = filter_peak_clusters(transitions60)

	# Step 6. construct transitions dataframe + write to csv
	transitions30['novelty_window'] = 30
	transitions60['novelty_window'] = 60
	transitions = pd.concat((transitions30,transitions60))
	outname = novelty_filepath.split('/')[-1].replace('-novelty','-transitions')
	transitions.to_csv(OUT_DIR+outname, index=False)
	return



##############################################
# MAIN
##############################################
infiles = glob(IN_DIR+"*")
for f in infiles:
	print(f)
	get_transitions(f)
