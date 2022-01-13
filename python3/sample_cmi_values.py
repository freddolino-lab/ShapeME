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
import numba
from numba import jit,prange
import pickle
import tempfile
import time
import subprocess
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/sample_cmi')

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

def seqs_per_bin(records):
    """ Function to determine how many sequences are in each category

    Args:
        records (SeqDatabase) - database to calculate over
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    string = ""
    for value in np.unique(records.y):
        string += "\nCat {}: {}".format(
            value, np.sum(records.y ==  value)
        )
    return string

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

@jit(nopython=True)
def calc_aic(delta_k, rec_num, mi):
    aic = 2*delta_k - 2*rec_num*mi
    return aic

def aic_motifs(motifs, records, optimized_vars):
    """Select final motifs through AIC

    Args:
        motifs (list of dicts) - list of final motif dictionaries
        records (SeqDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_motifs (list of dicts) - list of passing motif dictionaries
    """

    # get number of parameters based on window length * shape number * 2 + 1
    #  We multiply by 2 because for each shape value we have a weight,
    #  and we add 1 because the motif's threshold was a parameter.
    rec_num,win_len,shape_num,win_num,strand_num = records.windows.shape

    shape_num_multiplier = 0
    if 'shapes' in optimized_vars:
        shape_num_multiplier += 1
    if 'weights' in optimized_vars:
        shape_num_multiplier += 1

    delta_k = win_len * shape_num * shape_num_multiplier

    if 'threshold' in optimized_vars:
        delta_k += 1

    # sort motifs by mutual information
    these_motifs = sorted(
        motifs,
        key = lambda x: x['mi'],
        reverse = True,
    )

    # Make sure first motif passes AIC
    if calc_aic(delta_k, rec_num, these_motifs[0]['mi']) < 0:
        top_motif = these_motifs[0]
        top_motif['distinct_hits'] = np.unique(top_motif['hits'], axis=0)
        top_motifs = [these_motifs[0]]
    else:
        return []

    distinct_y = np.unique(records.y)

    # loop through candidate motifs
    for cand_motif in these_motifs[1:]:
        cand_hits = cand_motif['hits']
        distinct_cand_hits = np.unique(cand_hits, axis=0)
        motif_pass = True

        # if the total MI for this motif doesn't pass AIC skip it
        if calc_aic(delta_k, rec_num, cand_motif['mi']) > 0:
            continue

        for good_motif in top_motifs:
            # check the conditional mutual information for this motif with
            # each of the chosen motifs
            good_motif_hits = good_motif['hits']
            distinct_good_motif_hits = good_motif['distinct_hits']

            this_cmi = inout.conditional_adjusted_mutual_information(
                records.y, 
                #distinct_y,
                cand_hits, 
                #distinct_cand_hits,
                good_motif_hits,
                #distinct_good_motif_hits,
            )
            this_aic = calc_aic(delta_k, rec_num, this_cmi)

            # if candidate motif doesn't improve model as added to each of the
            # chosen motifs, skip it
            if this_aic > 0:
                motif_pass = False
                break

        if motif_pass:
            cand_motif['distinct_hits'] = distinct_cand_hits
            top_motifs.append(cand_motif)

    return top_motifs


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
    parser.add_argument('--init_threshold_seed_num', type=float, default=500.0, 
            help="Number of randomly selected seeds to compare to records in the database during initial threshold setting. Default=500.0")
    parser.add_argument('--init_threshold_recs_per_seed', type=float, default=20.0, 
            help="Number of randomly selected records to compare to each seed during initial threshold setting. Default=20.0")
    parser.add_argument('--init_threshold_windows_per_record', type=float, default=2.0, 
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

    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        #records.read(args.infile, float)
        #logging.info("Discretizing data")
        #records.discretize_quant(args.continuous)
        #logging.info("Quantizing input data using k-means clustering")
        records.quantize_quant(args.continuous)

    logging.info("Distribution of sequences per class:")
    logging.info(seqs_per_bin(records))

    logging.info("Normalizing parameters")
    if args.nonormalize:
        records.determine_center_spread(method=inout.identity_csp)
    else:
        records.determine_center_spread()
        records.normalize_shape_values()

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        logging.info(
            "{}: center={}, spread={}".format(
                name,
                this_center,
                this_spread
            )
        )

    alpha = args.alpha
    max_count = args.max_count
    optim_str = "shapes_weights_threshold"

    temp = args.temperature
    step = args.stepsize
    
    mi_fname = os.path.join(
        out_direc,
        '{}_initial_mutual_information_max_count_{}.pkl'.format(
            out_pref,
            max_count,
        ),
    )

    shape_fname = os.path.join(out_direc, 'shapes.npy')
    yval_fname = os.path.join(out_direc, 'y_vals.npy')
    config_fname = os.path.join(out_direc, 'config.pkl')
    rust_out_fname = os.path.join(out_direc, 'cmi_samples.pkl')
    dist_plot_out_fname = os.path.join(out_direc, 'distribution.png')
    fit_plot_out_fname = os.path.join(out_direc, 'fit.png')

    good_motif_out_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_optim_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.pkl".format(
            out_pref,
            optim_str,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    good_motif_plot_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_optim_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.png".format(
            out_pref,
            optim_str,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    RUST = "{} {} {} {} {}".format(
        rust_bin,
        shape_fname,
        yval_fname,
        config_fname,
        rust_out_fname,
    )

    args_dict = {
        'alpha': args.alpha,
        'max_count': float(args.max_count),
        'kmer': float(args.kmer),
        'cores': float(args.p),
        'seed_sample_size': float(args.init_threshold_seed_num),
        'records_per_seed': float(args.init_threshold_recs_per_seed),
        'windows_per_record': float(args.init_threshold_windows_per_record),
        'thresh_sd_from_mean': args.threshold_sd,
        'threshold_lb': args.threshold_constraints[0],
        'threshold_ub': args.threshold_constraints[1],
        'shape_lb': args.shape_constraints[0],
        'shape_ub': args.shape_constraints[1],
        'weight_lb': args.weights_constraints[0],
        'weight_ub': args.weights_constraints[1],
        'temperature': args.temperature,
        'stepsize': args.stepsize,
        'n_opt_iter': args.opt_niter,
        't_adjust': args.t_adj,
        'batch_size': args.batch_size,
    }

    # write shapes to npy file. Permute axes 1 and 2.
    with open(shape_fname, 'wb') as shape_f:
        np.save(shape_fname, records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, records.y.astype(np.int64))
    # write cfg to file
    with open(config_fname, 'wb') as f:
        pickle.dump(args_dict, f)

    logging.info("Sampling random pairs of seeds to approximate CMI distribution under null.")
    retcode = subprocess.call(RUST, shell=True)
    if retcode != 0:
        sys.exit("ERROR: CMI sampling binary had non-zero exit status")

    # remove the files used or generated by rust
    os.remove(shape_fname)
    os.remove(yval_fname)
    os.remove(config_fname)

    with open(rust_out_fname, 'rb') as f:
        data = np.asarray(pickle.load(f))
    sort_idx = np.argsort(data)
    data = data[sort_idx]
    # if min cmi val was less than 0, shift by -min and add epsilon
    if data.min() < 0:
        data -= data.min()
        data += sys.float_info.epsilon

    sns.displot(data)
    plt.savefig(dist_plot_out_fname)
    plt.close()

    #with open(os.path.join(out_direc, 'empirical_cdf.pkl'), 'rb') as f:
    #    ecdf = pickle.load(f)
    with open(os.path.join(out_direc, 'fitted_pdf.pkl'), 'rb') as f:
        fitted_pdf = np.asarray(pickle.load(f))[sort_idx]

    sns.displot(data, kind='kde')
    plt.plot(data, fitted_pdf, 'r-')
    plt.savefig(fit_plot_out_fname)
    plt.close()

