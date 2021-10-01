from importlib import reload
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
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import cvlogistic
from scipy.stats import sem
from scipy import optimize as opt

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

alpha = float(sys.argv[2])

this_dist = inout.constrained_inv_logit_manhattan_distance

in_file = "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
kmer = 15
thresh_sd_from_mean = 2.0
max_count = int(sys.argv[1])

print("Reading in data")
shape_fname_dict = {n:fn for n,fn in zip(param_names, params)}
rec_db = inout.RecordDatabase(in_file, shape_fname_dict)
rec_db.determine_center_spread()
rec_db.normalize_shape_values()
rec_db.compute_windows(wsize=kmer)
weights = rec_db.initialize_weights()
# add new axis to broadcast later in distance calculation
weights = weights[...,None]
print("Setting initial guess at appropriate threshold for matches")
threshold = rec_db.set_initial_threshold(
    dist = this_dist,
    weights = weights,
    threshold_sd_from_mean = thresh_sd_from_mean,
    alpha = alpha,
)

print("Computing mutual information for all seeds")
start = time.time()
mi_results = rec_db.compute_mi(
    dist = this_dist,
    max_count = max_count,
    alpha = alpha,
    weights = weights,
    threshold = threshold,
)
end = time.time()
out_dict = {
    'mi_results': mi_results,
    'weights': weights,
    'threshold': threshold,
    'max_count': max_count,
    'alpha': alpha,
}

print("Time elapsed: {} minutes".format((end-start)/60))
with open(
    '{}_initial_mutual_information_alpha_{}_max_count_{}.pkl'.format(
        os.path.splitext(in_file)[0],
        alpha,
        max_count
    ),
    'wb'
) as f:
    pickle.dump(out_dict, f)

