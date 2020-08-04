#!/bin/bash

echo Running Librosa-Pipeline.

echo Get MFCC from audio
python freejazz-preprocessing-mfcc.py

echo Compute self similarity matrix
python freejazz_mfcc_similarity.py

echo Compute novelty time series
python freejazz_novelty.py

echo Detect Transitions
python freejazz_transitions.py

echo Compute CSD signals
python freejazz_csd.py

echo Compute Kendalls Tau
python freejazz_tau.py
