####################################################
# Compute Kendall's tau for CSD indices
#
# MS 8/22/19
####################################################

import numpy as np
import pandas as pd
from glob import glob


IN_DIR = '../Librosa-Pipeline/csd/'
OUT_DIR = '../Librosa-Pipeline/csd-tau/'
# STEP_SIZE = 0.2  (inherit this from csd time series)


##############################################
# FUNCTIONS
##############################################

def compute_tau_vals(csd_df, tau_window):
	acf_taus = []
	var_taus = []
	for time in csd_df['t']:
		csd_t = csd_df.loc[(csd_df.t>(time-tau_window))&(csd_df.t<=time)]
		acf_taus.append(csd_t['t'].corr(csd_t['acf'],method='kendall'))
		var_taus.append(csd_t['t'].corr(csd_t['var'],method='kendall'))

	return (acf_taus, var_taus)


def get_kendalls_tau(csd_filepath):
	print("get_kendalls_tau")

	# read in csd time series
	csd = pd.read_csv(csd_filepath)

	# for each window, compute kendall's tau for var and acf
	tau_df = pd.DataFrame()
	tau_windows = [5,10,30]
	for tau_window in tau_windows:
		print(tau_window)
		for w in np.unique(csd['window']):
			csd_w = csd.loc[csd.window == w]
			acf_taus, var_taus = compute_tau_vals(csd_w, tau_window)
			tau_df = pd.concat((tau_df, pd.DataFrame({'t': csd_w['t'],'acf_tau': acf_taus,
													  'var_tau': var_taus, 'csd_w': w, 
													  'tau_w': tau_window})))

	# write out dataframe
	outname = csd_filepath.split('/')[-1].replace('-csd','-tau')
	tau_df.to_csv(OUT_DIR+outname, index = False)
	return



##############################################
# MAIN
##############################################
infiles = glob(IN_DIR+"*")
completed = [s.split('/')[-1].replace('-tau.csv','') for s in glob(OUT_DIR+"*")]
infiles = [f for f in infiles if not f.split('/')[-1].replace('-csd.csv','') in completed]
print(len(infiles))
for f in infiles:
	print(f)
	try:
		get_kendalls_tau(f)
	except:
		print('error processing '+f)
