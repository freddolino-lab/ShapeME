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
search_method = "brute"
in_file = "test/BRCA1_input/BRCA1_30_bp_height_train_4.txt"
kmer = 15
ignorestart = 2
ignoreend = 2
threshold_perc = 0.05
thresh_sd_from_mean = 2.0
threshold_seeds = 2.0
threshold_match = 2.0
seeds_per_seq_thresh = 1
seeds_per_seq = 2
num_seeds = 5000
rc = False
numprocs = 24
numba.set_num_threads(numprocs)
shape_fname_dict = {n:fn for n,fn in zip(param_names, params)}
rec_db = inout.RecordDatabase(in_file, shape_fname_dict)
rec_db.determine_center_spread()
rec_db.normalize_shape_values()
rec_db.compute_windows(wsize=15)
rec_db.initialize_weights()
rec_db.set_initial_thresholds(
    dist = this_dist,
    threshold_sd_from_mean = thresh_sd_from_mean
)
start = time.time()
print(start)
rec_db.compute_mi(this_dist)
end = time.time()
print(end)
print("Time elapsed: {} minutes".format((end-start)/60))
with open('initial_mutual_information.pkl','wb') as f:
    pickle.dump(rec_db.mi, f)


#with open('rec_subset.pkl','rb') as f:
#    rec_subset = pickle.load(f)
#print("Running optimization")
#results = fm.mp_optimize_weights(
#    rec_db,
#    inout.constrained_manhattan_distance,
#    rec_subset[0:2],
#    p = numprocs,
#)
#print("Finished optimization of weights, gathering results")
#final_results = []
#for res in results:
#    final_results.append(res.get())
#print("Writing results to file")
#with open("test_subset_optim.pkl", "wb") as f:
#    pickle.dump(dict_results, f)
#print(time.time())