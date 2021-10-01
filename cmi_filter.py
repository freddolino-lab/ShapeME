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

in_file = "synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"

alpha = float(sys.argv[2])
max_count = int(sys.argv[1])

out_pref = os.path.splitext(in_file)[0]

this_dist = inout.constrained_inv_logit_manhattan_distance

kmer = 15
thresh_sd_from_mean = 2.0

shape_fname_dict = {n:fn for n,fn in zip(param_names, params)}
rec_db = inout.RecordDatabase(in_file, shape_fname_dict)
rec_db.determine_center_spread()
rec_db.normalize_shape_values()
rec_db.compute_windows(wsize=15)
rec_db.initialize_weights()
rec_db.set_initial_thresholds(
    dist = this_dist,
    threshold_sd_from_mean = thresh_sd_from_mean,
    alpha = alpha,
)

with open(
    '{}_initial_mutual_information_alpha_{}_max_count_{}.pkl'.format(
        out_pref,
        alpha,
        max_count
    ),
    'rb'
) as f:
    d = pickle.load(f)
    rec_db.mi = d['mi']
    rec_db.hits = d['hits']
    this_max_count = d['max_count']
    this_alpha = d['alpha']

if this_max_count != max_count:
    sys.exit("max_count set at CLI is not equal to max_count in pickle file!")
if this_alpha != alpha:
    sys.exit("alpha set at CLI is not equal to alpha in pickle file!")

print("Filtering seeds based on conditional mutual information")
cmi_windows = fm.aic_motifs2(rec_db)

with open(
    '{}_cmi_keeper_indices_alpha_{}_max_count_{}.pkl'.format(
        out_pref,
        alpha,
        max_count
    ),
    'wb'
) as outf:
    pickle.dump(cmi_windows, outf)

print("{} motifs to be optimized after CMI filter.".format(len(cmi_windows)))

