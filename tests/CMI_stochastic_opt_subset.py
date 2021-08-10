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
from scipy import stats

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
numprocs = 12

numba.set_num_threads(numprocs)

params_for_opt = sys.argv[1:]
if not params_for_opt:
    sys.exit("ERROR: you must pass any combination of {{weights, threshold, shapes}} to {}.".format(sys.argv[0]))

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

with open('fwd_rev_initial_mutual_information.pkl','rb') as f:
    d = pickle.load(f)
    rec_db.mi = d['mi']
    rec_db.hits = d['hits']

print("Filtering seeds based on conditional mutual information")
cmi_windows = fm.aic_motifs2(rec_db)

with open('cmi_keeper_indices.pkl','wb') as outf:
    pickle.dump(cmi_windows, outf)

#with open('cmi_keeper_indices.pkl','rb') as cmif:
#    cmi_windows = pickle.load(cmif)

print("{} motifs to be optimized after CMI filter.".format(len(cmi_windows)))

raise()

print("Optimizing {}".format(params_for_opt))
max_evals = None

temp = 0.5
step = 0.5
method = "nelder-mead"
fatol = 0.001
adapt = False

print("Running optimization")
start_time = time.time()
final_results = fm.stochastic_optimize(
    rec_db,
    inout.constrained_manhattan_distance,
    window_inds = cmi_windows,
    opt_params = params_for_opt,
    maxfev = max_evals,
    fatol = fatol,
    adapt = adapt,
    temp = temp,
    stepsize = step,
    method = method,
)
end_time = time.time()
print("Finished optimization of weights.")

print("Writing results to file")
optim_str = "_".join(params_for_opt)
with open(
    "stochastic_opt_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_min_mi_{:.1f}.pkl".format(
        optim_str,
        adapt,
        fatol,
        temp,
        step,
        min_mi,
    ),
    "wb",
) as f:
    pickle.dump(final_results, f)
print("Time for optimization: {:.2f} minutes".format((end_time-start_time)/60))
