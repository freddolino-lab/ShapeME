from importlib import reload
import pathlib
import os
import sys
import time
import pickle
from pprint import pprint
import numba
from numba import jit,prange
this_path = pathlib.Path('.').parent.absolute()
src_path = os.path.join(this_path, '..')
sys.path.insert(0, src_path)
import dnashapeparams as dsp
import find_motifs as fm
import numpy as np
import inout
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import cvlogistic
from scipy.stats import sem
from scipy import optimize as opt
params = [
    "test/BRCA1_input/BRCA1_30_bp_height_train_4.fa.EP",
    "test/BRCA1_input/BRCA1_30_bp_height_train_4.fa.HelT",
    "test/BRCA1_input/BRCA1_30_bp_height_train_4.fa.MGW",
    "test/BRCA1_input/BRCA1_30_bp_height_train_4.fa.ProT",
    "test/BRCA1_input/BRCA1_30_bp_height_train_4.fa.Roll",
]
param_names = [
    "EP",
    "HelT",
    "MGW",
    "ProT",
    "Roll",
]

this_dist = inout.constrained_manhattan_distance
in_file = "test/BRCA1_input/BRCA1_30_bp_height_train_4.txt"
kmer = 15
thresh_sd_from_mean = 2.0
numprocs = 6

numba.set_num_threads(numprocs)

print("Reading in data")
shape_fname_dict = {n:fn for n,fn in zip(param_names, params)}
rec_db = inout.RecordDatabase(in_file, shape_fname_dict)
rec_db.determine_center_spread()
rec_db.normalize_shape_values()
rec_db.compute_windows(wsize=kmer)
rec_db.initialize_weights()
print("Setting initial guess at appropriate threshold for matches")
rec_db.set_initial_thresholds(
    dist = this_dist,
    threshold_sd_from_mean = thresh_sd_from_mean
)

print("Computing mutual information for all seeds")
start = time.time()
rec_db.compute_mi(this_dist, binary=True)
end = time.time()
out_dict = {
    'mi': rec_db.mi,
    'hits': rec_db.hits,
}

print("Time elapsed: {} minutes".format((end-start)/60))
with open('binary_fwd_rev_initial_mutual_information.pkl','wb') as f:
    pickle.dump(out_dict, f)

