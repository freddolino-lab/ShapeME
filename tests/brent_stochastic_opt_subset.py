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
numprocs = 4

numba.set_num_threads(numprocs)

params_for_opt = sys.argv[1:]
if not params_for_opt:
    sys.exit("ERROR: you must pass any combination of {{weights, shapes}} to {}.".format(sys.argv[0]))

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
    rec_db.mi = pickle.load(f)

# We'll set the min MI to the 85% percentile here
min_mi = stats.scoreatpercentile(rec_db.mi, 85)

# grab 40 evenly-spaced values within this top 15%
quant_vals = stats.scoreatpercentile(
    rec_db.mi[rec_db.mi >= min_mi],
    np.linspace(0,100,2),
    interpolation_method = 'lower',
)

row_inds = []
col_inds = []

for qval in quant_vals:
    qinds = np.where(rec_db.mi == qval)
    row_inds.append(qinds[0][0])
    col_inds.append(qinds[1][0])

windows_for_opt = (row_inds, col_inds)

print("Optimizing {}".format(params_for_opt))
max_evals = None

temp = 0.5
step = 0.5
method = "nelder-mead"
fatol = 0.01
adapt = False

print("Running optimization")
start_time = time.time()

fm.brent_optimize_threshold(
    rec_db,
    inout.constrained_manhattan_distance,
    window_inds = windows_for_opt,
)

final_results = fm.stochastic_optimize(
    rec_db,
    inout.constrained_manhattan_distance,
    window_inds = windows_for_opt,
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
    "brent_stochastic_opt_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_min_mi_{:.1f}.pkl".format(
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
