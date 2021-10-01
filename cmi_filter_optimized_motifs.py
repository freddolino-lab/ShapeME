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
    "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.fa.EP",
    "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.fa.HelT",
    "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.fa.MGW",
    "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.fa.ProT",
    "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.fa.Roll",
]
param_names = [
    "EP",
    "HelT",
    "MGW",
    "ProT",
    "Roll",
]

alpha = float(sys.argv[1])

this_dist = inout.constrained_inv_logit_manhattan_distance

in_file = "synthetic_data_2000_recs_frac_0.2_sesqlen_60_TBX5motif_TCTCACACCT.txt"
kmer = 15
thresh_sd_from_mean = 2.0

print(sys.argv)

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

temp = 0.10
step = 0.25
method = "nelder-mead"
fatol = 0.001
adapt = False
max_count = 1

print("Reading optimization results from file")
with open(
    "stochastic_opt_weights_shapes_threshold_adapt_{}_fatol_{}_temp_{}_stepsize_{}_max_count_{}_alpha_{}.pkl".format(
        adapt,
        fatol,
        temp,
        step,
        max_count,
        alpha,
    ),
    "rb",
) as f:
    optim_results = pickle.load(f)

optim_results = [opt[0] for opt in optim_results]
filtered_motifs = fm.aic_motifs3(optim_results, rec_db)

with open(
    "filtered_optimized_motifs_temp_{}_stepsize_{}_max_count_{}_alpha_{}.pkl".format(
        temp,
        step,
        max_count,
        alpha,
    ),
    "wb",
) as f:
    pickle.dump(filtered_motifs, f)

