import inout
import glob
import sys
import os
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
import shapemotifvis as smv
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import itertools
import welfords
import sklearn
import cvlogistic
import numba
from numba import jit,prange
import pickle
import json
import tempfile
import time
import subprocess
from pathlib import Path

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/find_motifs')


def generate_dist_vector(data, motif, rc=False):
    """ Function to calculate the best possible match value for each seq
    Args:
        data (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
        motif (np.array) - numpy array containing motif vector
        rc (bool) - check the reverse complement matches as well
    Returns:
        discrete (np.array) - a numpy array of minimum match value for each seq
    """
    all_matches = []
    motif_vec = motif.as_vector(cache=True)
    if rc:
        motif.rev_comp()
        motif_vec_rc = motif.as_vector()
        motif.rev_comp()
    this_seq_matches = []
    for this_seq in data.iterate_through_precompute():
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            this_seq_matches.append(distance)
        if rc:
            for this_motif in this_seq:
                distance = this_motif.distance(motif_vec_rc, vec=True, cache=True)
                this_seq_matches.append(distance)

        all_matches.append(np.min(this_seq_matches))
        this_seq_matches = []
    return np.array(all_matches)

def two_way_to_log_odds(two_way):
    """ Function to determine the log odds from a two way table

    Args:
        two_way (list)- a list of 0-cat1Truecat2True 1-cat1Falsecat2True 
                                  2-cat1Truecat2False 3-cat1Falsecat2False
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    num = np.array(two_way[0], dtype=float) / np.array(two_way[1],dtype=float)
    denom = np.array(two_way[2], dtype=float) / np.array(two_way[3],dtype=float)
    return np.log(num/denom)

def read_parameter_file(infile):
    """ Wrapper to read a single parameter file

    Args:
        infile (str) - input file name
    Returns:
        inout.FastaFile object containing data
    """
    fastadata = inout.FastaFile()
    with open(infile) as f:
        fastadata.read_whole_datafile(f)
    return fastadata

def print_top_motifs(motifs, n= 5, reverse=True):
    """
    Function to print the top motifs sorted by MI

    Args
        motifs (list) - motifs to sort
        n (int) - number of motifs to print
        reverse (bool) - sort in reverse
    Modifys
        stdout through the logging function
    """

    sorted_motifs = sorted(
        motifs,
        key=lambda x: x['mi'],
        reverse=reverse,
    )
    if reverse:
        logging.debug("Printing top {} motifs.".format(n))
    else:
        logging.debug("Printing bottom {} motifs.".format(n))

    for motif in sorted_motifs[0:n]:
        logging.debug("Motif MI: {}\n Motif Mem: {}\n{}".format(
            motif['mi'],
            motif['motif'],
            motif['motif'].as_vector(),
        ))

def info_zscore(vec1, vec2, n=10000):
    """ Similar to FIRE determine a Z score for vec1 based on MI scores
    from randomly shuffled vec2

    Args:
        vec1 - vector to not shuffle
        vec2 -vector to shuffle
        n - number of shuffles to do
    
    Returns:
        zscore - (MI_actual - mean MI_shuff)/std
        passed - was MI_actual greater than all shuffles?
    """
    # doing welford's again
    online_mean = welfords.Welford()
    passed = True
    actual = inout.mutual_information(vec1, vec2)
    for i in range(n):
        shuffle = np.random.permutation(len(vec2))
        newval = inout.mutual_information(vec1, vec2[shuffle])
        online_mean.update(newval)
        if newval >= actual:
            passed = False
    mean = online_mean.final_mean()
    stdev = online_mean.final_stdev()
    zscore = (actual-mean)/stdev
    return zscore, passed

def info_robustness(vec1, vec2, n=10000, r=10, holdout_frac=0.3):
    """ Similar to FIRE Robustness score, calculate Z score for
    jacknife replicates and report number that pass

    Args:
        vec1 - vector to not shuffle
        vec2 -vector to shuffle
        n - number of shuffles to do for zscore
        r - number of jackknife replicates
        holdout_frac - fraction of data to remove for jackknife
    
    Returns:
        num_passed - number of jacknife reps that passed
    """
    num_passed = 0
    num_to_use = int(np.floor((1-holdout_frac)*len(vec1)))
    for i in range(r):
        jk_selector = np.random.permutation(len(vec1))[0:num_to_use]
        zscore, passed = info_zscore(vec1[jk_selector], vec2[jk_selector], n=n)
        if passed:
            num_passed += 1
    return num_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', action='store', type=str, required=True,
        help='input text file with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help="Sets the upper and lower limits on the match threshold during optimization. Defaults to 0 for the lower limit and 10 for the upper limit.")
    parser.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the shapes' z-scores during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, pre-normalized weights during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--temperature', type=float, default=0.1,
        help="Sets the temperature argument for scipy.optimize.basinhopping")
    parser.add_argument('--t_adj', type=float, default=0.001,
        help="Fraction by which temperature decreases each iteration ofsimulated annealing.")
    parser.add_argument('--stepsize', type=float, default=0.25,
        help="Sets the stepsize argument for scipy.optimize.basinhopping")
    parser.add_argument('--opt_niter', type=int, default=100,
        help="Sets the number of simulated annealing iterations to undergo during optimization. Defaults to 100.")
    parser.add_argument('--kmer', type=int,
                         help='kmer size to search for. Default=15', default=15)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--threshold_sd', type=float, default=2.0, 
            help="std deviations below mean for seed finding. Only matters for greedy search. Default=2.0")
    parser.add_argument('--init_threshold_seed_num', type=int, default=500, 
            help="Number of randomly selected seeds to compare to records in the database during initial threshold setting. Default=500.0")
    parser.add_argument('--init_threshold_recs_per_seed', type=int, default=20, 
            help="Number of randomly selected records to compare to each seed during initial threshold setting. Default=20.0")
    parser.add_argument('--init_threshold_windows_per_record', type=int, default=2, 
            help="Number of randomly selected windows within a given record to compare to each seed during initial threshold setting. Default=2.0")
    parser.add_argument('--motif_perc', type=float, default=1,
            help="fraction of data to EVALUATE motifs on. Default=1")
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--alpha', type=float, default=0.0,
            help="Lower limit on transformed weight values prior to normalization to sum to 1. Defaults to 0.0.")
    parser.add_argument('--max_count', type=int, default=1,
            help="Maximum number of times a motif can match each of the forward and reverse strands in a reference.")
    #parser.add_argument('--infoz', type=int, default=2000, 
    #        help="Calculate Z-score for final motif MI with n data permutations. default=2000. Turn off by setting to 0")
    #parser.add_argument('--inforobust', type=int, default=10, 
    #        help="Calculate robustness of final motif with x jacknifes. Default=10. Requires infoz to be > 0.")
    #parser.add_argument('--fracjack', type=int, default=0.3, 
    #        help="Fraction of data to hold out in jacknifes. Default=0.3.")
    parser.add_argument('-o', type=str, required=True, help="Prefix to apply to output files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory from which input files will be read.")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory (within 'data_dir') into which output files will be written.")
    parser.add_argument('-p', type=int, default=5,
        help="number of processors. Default=5")
    parser.add_argument('--batch_size', type=int, default=2000,
        help="Number of records to process seeds from at a time. Set lower to avoid out-of-memory errors. Default=2000")
    #parser.add_argument("--debug", action="store_true",
    #    help="print debugging information to stderr. Write extra txt files.")
    #parser.add_argument('--txt_only', action='store_true', help="output only txt files?")
    #parser.add_argument('--save_opt', action='store_true', help="write motifs to pickle file after initial weights optimization step?")

    my_env = os.environ.copy()
    my_env['RUST_BACKTRACE'] = "1"
    
    args = parser.parse_args()
    numba.set_num_threads(args.p)
    out_pref = args.o
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level, stream=sys.stdout) 
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.info("Reading in files")
    # read in shapes
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(args.param_names, args.params)
    }
    logging.info("Reading input data and shape info.")
    records = inout.RecordDatabase(
        os.path.join(in_direc, args.infile),
        shape_fname_dict,
        shift_params = ["Roll", "HelT"],
    )

