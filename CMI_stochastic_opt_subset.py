import pathlib
import os
import sys
import time
import pickle
from pprint import pprint
import numba
from numba import jit,prange
src_path = os.path.join(os.environ["HOME"], 'src/DNAshape_motif_finder')
sys.path.insert(0, src_path)
import dnashapeparams as dsp
import find_motifs as fm
import numpy as np
import inout
from scipy import stats

params = [
    "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.EP",
    "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.HelT",
    "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.MGW",
    "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.ProT",
    "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.Roll",
]
param_names = [
    "EP",
    "HelT",
    "MGW",
    "ProT",
    "Roll",
]

this_dist = inout.constrained_inv_logit_manhattan_distance

alpha = float(sys.argv[3])
in_file = "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
kmer = 15
thresh_sd_from_mean = 2.0

params_for_opt = sys.argv[4:]
if not params_for_opt:
    sys.exit("ERROR: you must pass any combination of {{weights, threshold, shapes}} to {}.".format(sys.argv[0]))

shape_fname_dict = {n:fn for n,fn in zip(param_names, params)}
rec_db = inout.RecordDatabase(in_file, shape_fname_dict)
rec_db.determine_center_spread()
rec_db.normalize_shape_values()
rec_db.compute_windows(wsize=kmer)
rec_db.initialize_weights()
rec_db.set_initial_thresholds(
    dist = this_dist,
    threshold_sd_from_mean = thresh_sd_from_mean,
    alpha = alpha,
)

out_pref = os.path.splitext(in_file)[0]

with open(
    '{}_initial_mutual_information_alpha_{}_max_count_{}.pkl'.format(
        out_pref,
        alpha,
        sys.argv[2],
    ),
    'rb'
) as f:
    d = pickle.load(f)
    rec_db.mi = d['mi']
    rec_db.hits = d['hits']
    max_count = d['max_count']
    alpha = d['alpha']

windows_file = sys.argv[1]

with open(windows_file, 'rb') as cmif:
    cmi_windows = pickle.load(cmif)

windows_direc,windows_base = os.path.split(windows_file)
print(windows_direc)
print(windows_base)
optim_direc = os.path.join(
    windows_direc,
)

#if not os.path.isdir(optim_direc):
#    os.makedirs(optim_direc)

print("Optimizing {} for {} seeds.".format(params_for_opt, len(cmi_windows)))
max_evals = None

temp = 0.1
step = 0.25
method = "nelder-mead"
fatol = 0.01
adapt = False
niter = 100
niter_success = None
print("Using lower limit on transformed weights of {}.".format(alpha))
constraints = {
    'threshold': (0.0,10.0),
    'shapes': (-4,4),
    'weights': (-4,4),
}

optim_str = "_".join(params_for_opt)

out_fname = "{}_stochastic_opt_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.pkl".format(
    os.path.splitext(windows_base)[0],
    optim_str,
    adapt,
    fatol,
    temp,
    step,
    alpha,
    max_count,
)
out_fname = os.path.join(
    optim_direc,
    out_fname,
)
print(out_fname)

print("Running optimization")
start_time = time.time()
fm.stochastic_optimize(
    out_fname,
    rec_db,
    inout.constrained_inv_logit_manhattan_distance,
    window_inds = cmi_windows,
    opt_params = params_for_opt,
    maxfev = max_evals,
    fatol = fatol,
    adapt = adapt,
    temp = temp,
    stepsize = step,
    method = method,
    constraints_dict = constraints,
    alpha = alpha,
    max_count = max_count,
    niter = niter,
    niter_success = niter_success,
)
end_time = time.time()
print("Finished optimization of weights.")

#print("Writing results to file")
#optim_str = "_".join(params_for_opt)
#with open(
#    "{}_stochastic_opt_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_max_count_{}_alpha_{}.pkl".format(
#        out_pref,
#        optim_str,
#        adapt,
#        fatol,
#        temp,
#        step,
#        max_count,
#        alpha,
#    ),
#    "wb",
#) as f:
#    pickle.dump(final_results, f)
print("Time for optimization: {:.2f} minutes".format((end_time-start_time)/60))
