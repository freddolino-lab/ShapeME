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
from scipy import stats
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
numprocs = 12
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
#start = time.time()
#print(start)
#rec_db.compute_mi(this_dist)
#end = time.time()
#print(end)
#print("Time elapsed: {} minutes".format((end-start)/60))
with open('initial_mutual_information.pkl','rb') as f:
    rec_db.mi = pickle.load(f)

min_mi = 0.2

quant_vals = stats.scoreatpercentile(
    rec_db.mi[rec_db.mi >= min_mi],
    np.linspace(0,100,40),
    interpolation_method = 'lower',
)

row_inds = []
col_inds = []

for qval in quant_vals:
    qinds = np.where(rec_db.mi == qval)
    #print(qinds)
    row_inds.append(qinds[0][0])
    col_inds.append(qinds[1][0])

windows_for_opt = (row_inds, col_inds)

params_for_opt = sys.argv[1:]
print("Optimizing {}".format(params_for_opt))
max_evals = None

print("Running optimization")
start_time = time.time()
final_results = fm.mp_optimize(
    rec_db,
    inout.constrained_manhattan_distance,
    window_inds = windows_for_opt,
    opt_params = params_for_opt,
    maxfev = max_evals,
)
end_time = time.time()
print("Finished optimization of weights.")
#final_results = []
#for res in results:
#    final_results.append(res.get())
print("Writing results to file")
optim_str = "_".join(params_for_opt)
with open(
    "test_subset_optim_{}_flat_init_w_min_mi_{}.pkl".format(optim_str,min_mi),
    "wb",
) as f:
    pickle.dump(final_results, f)
print("Time for optimization: {:.2f} minutes".format((end_time-start_time)/60))
