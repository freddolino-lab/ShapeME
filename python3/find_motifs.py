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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import cvlogistic
import numba
from numba import jit,prange
import pickle
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
    parser.add_argument('--mi_file', action="store_true",
        help='Apply this flag if there is a file containing pre-computed mutual information for each seed.')
    parser.add_argument('--cmi_file', action="store_true",
        help='Apply this flag if there is a file containing CMI-filtered seeds. NOTE: If this flag is set, initial mutual informations will neither be calculated nor read in, even if you have set the --mi_file flag.')
    parser.add_argument('--optim_file', action="store_true",
        help='Apply this flag if there is a file containing optimized motifs that you would like to read ing. NOTE: If this flag is set, initial mutual informations will neither be calculated nor read in, even if you have set the --mi_file flag. Also, cmi-filtered seeds will not be read in, even if you have set the --cmi_file flag.')
    parser.add_argument('--cmi_motif_file', action="store_true",
        help='Apply this flag if there is a file containing CMI-filtered optimized motifs. NOTE: If this flag is set, initial MI calculation, CMI filtering of seeds, and optimization will be skipped. Pre-computed MIs, CMI-filtered seeds, and optimizations will not be read in, no matter whether you set the --mi_file, --cmi_file, or --optim_file flags.')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--no_optimize', action="store_true",
        help="Set this if skipping optimization")
    parser.add_argument('--optim_vars', nargs="+", type=str,
        default=["weights","threshold"],
        help="Names of the variables to optimize. Should be any combination of {'shapes', 'weights', 'threshold'}.")
    parser.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help="Sets the upper and lower limits on the match threshold during optimization. Defaults to 0 for the lower limit and 10 for the upper limit.")
    parser.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the shapes' z-scores during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, pre-normalized weights during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--temperature', type=float, default=0.1,
        help="Sets the temperature argument for scipy.optimize.basinhopping")
    parser.add_argument('--stepsize', type=float, default=0.25,
        help="Sets the stepsize argument for scipy.optimize.basinhopping")
    parser.add_argument('--basinhop_niter', type=int, default=100,
        help="Sets the number of basin hops to undergo during optimization. Defaults to 100.")
    parser.add_argument('--basinhop_niter_success', type=int, default=None,
        help="Sets the number of basin hops after which if the global minimum hasn't been improve, optimization terminates. Defaults to None, which means that the value set by basinhop_niter will be the sole determinant of how many hops will be carried out")
    parser.add_argument('--fatol', type=float, default=0.01,
        help="Sets the fatol argument to control convergence of the Nelder-Mead local optimizer.")
    parser.add_argument('--maxfev', type=int, default=None,
        help="Sets the maxfev argument to control the Nelder-Mead local optimizer.")
    parser.add_argument('--adapt', action="store_true",
        help="Applying this flag will set adapt=True for the Nelder-Mead optimizer")
    parser.add_argument('--kmer', type=int,
                         help='kmer size to search for. Default=15', default=15)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--threshold_perc', type=float, default=0.1,
            help="fraction of data to determine threshold on. Default=0.1")
    parser.add_argument('--threshold_seeds', type=float, default=2.0, 
            help="std deviations below mean for seed finding. Only matters for greedy search. Default=2.0")
    parser.add_argument('--threshold_match', type=float, default=2.0, 
            help="std deviations below mean for match threshold. Default=2.0")
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
    parser.add_argument('--mi_perc', type=float, default=5,
            help="ratio of CMI/MI to include an additional motif. Default=5")
    parser.add_argument('--alpha', type=float, default=0.0,
            help="Lower limit on transformed weight values prior to normalization to sum to 1. Defaults to 0.0.")
    parser.add_argument('--max_count', type=int, default=1,
            help="Maximum number of times a motif can match each of the forward and reverse strands in a reference.")
    parser.add_argument('--infoz', type=int, default=2000, 
            help="Calculate Z-score for final motif MI with n data permutations. default=2000. Turn off by setting to 0")
    parser.add_argument('--inforobust', type=int, default=10, 
            help="Calculate robustness of final motif with x jacknifes. Default=10. Requires infoz to be > 0.")
    parser.add_argument('--fracjack', type=int, default=0.3, 
            help="Fraction of data to hold out in jacknifes. Default=0.3.")
    parser.add_argument('--distance_metric', type=str, default="constrained_manhattan",
            help="distance metric to use, manhattan using constrained weights is the only supported one for now")
    parser.add_argument('--seed', type=int, default=None,
            help="set the random seed, default=None, based on system time")
    parser.add_argument('-o', type=str, required=True, help="Prefix to apply to output files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory from which input files will be read.")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory (within 'data_dir') into which output files will be written.")
    parser.add_argument('-p', type=int, default=5,
        help="number of processors. Default=5")
    parser.add_argument("--exit_after_initial_mi", action="store_true",
        help="Run initial mutual information calculation, then exit, saving the result of the mutual information calculations.")
    parser.add_argument("--exit_after_cmi_motifs", action="store_true",
        help="Exit after filtering optmized motifs by their CMI.")
    parser.add_argument("--exit_after_cmi_filter", action="store_true",
        help="Run initial CMI filtering step prior to optimization, then exit, saving the retained seeds as a list in a pickle file.")
    parser.add_argument("--exit_after_optimization", action="store_true",
        help="Run initial CMI filtering step prior to optimization, then exit, saving the retained seeds as a list in a pickle file.")
    parser.add_argument("--debug", action="store_true",
        help="print debugging information to stderr. Write extra txt files.")
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

    optim_vars = args.optim_vars
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level) 
    logging.getLogger('matplotlib.font_manager').disabled = True

    # choose a random seed
    if args.seed:
        np.random.seed(args.seed)
    
    logging.info("Reading in files")
    # possible distance metrics that could be used
    dist_met = {"constrained_manhattan": inout.constrained_inv_logit_manhattan_distance,
                "manhattan": inout.manhattan_distance, 
                "hamming": inout.hamming_distance,
                "euclidean": inout.euclidean_distance}
    # store the distance metric chosen
    this_dist = dist_met[args.distance_metric]
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

    optim_str = "_".join(optim_vars)
    temp = args.temperature
    step = args.stepsize
    fatol = args.fatol
    adapt = args.adapt
    maxfev = args.maxfev
    
    constraints = {
        'threshold': args.threshold_constraints,
        'shapes': args.shape_constraints,
        'weights': args.weights_constraints,
    }

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
    rust_out_fname = os.path.join(out_direc, 'rust_results.pkl')

    cmi_fname = os.path.join(
        out_direc,
        "{}_cmi_filtered_seeds_opim_{}_max_count_{}.pkl".format(
            out_pref,
            optim_str,
            max_count,
        ),
    )

    opt_direc = os.path.join(out_direc, "optimizations")

    opt_fname = os.path.join(
        opt_direc,
        "{}_optim_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}_batch_{{:0=3}}.pkl".format(
            out_pref,
            optim_str,
            adapt,
            fatol,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    good_motif_out_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_optim_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.pkl".format(
            out_pref,
            optim_str,
            adapt,
            fatol,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    good_motif_plot_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_optim_{}_adapt_{}_fatol_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.png".format(
            out_pref,
            optim_str,
            adapt,
            fatol,
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
        'thresh_sd_from_mean': args.threshold_seeds,
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

    
    logging.info("Running motif selection and optimization.")
    retcode = subprocess.call(RUST, shell=True)
    if retcode != 0:
        sys.exit("ERROR: motifer binary execution exited with non-zero exit status")

    good_motifs = inout.read_motifs_from_rust(rust_out_fname)

    # remove the files used or generated by rust
    os.remove(shape_fname)
    os.remove(yval_fname)
    os.remove(config_fname)
    os.remove(rust_out_fname)

    with open(good_motif_out_fname, 'wb') as outf:
        pickle.dump(good_motifs, outf)

    smv.plot_optim_shapes_and_weights(good_motifs, good_motif_plot_fname, records)

    raise()

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

    X = [
        this_motif['dists']
        for this_motif in good_motifs
    ]
    X = np.stack(X, axis=1)
    X = StandardScaler().fit_transform(X)
    y = this_records.get_values()
    logging.info("Running L1 regularized logistic regression with CV to determine reg param")

    clf = LogisticRegressionCV(
        Cs=100,
        cv=5,
        multi_class='multinomial',
        penalty='l1',
        solver='saga',
        max_iter=10000,
    ).fit(X, y)

    best_c = cvlogistic.find_best_c(clf)

    clf_f = LogisticRegression(
        C=best_c,
        multi_class='multinomial',
        penalty='l1',
        solver='saga',
        max_iter=10000,
    ).fit(X,y)

    good_motif_index = cvlogistic.choose_features(clf_f, tol=0)
    if len(good_motif_index) < 1:
        logging.info("No motifs found")
        sys.exit()

    cvlogistic.write_coef_per_class(clf_f, outpre + "_coef_per_class.txt")
    final_good_motifs = [good_motifs[index] for index in good_motif_index]
    logging.info("{} motifs survived".format(len(final_good_motifs)))

    raise()

########################################################################
    if args.debug:
        cvlogistic.plot_score(clf, outpre+"_score_logit.png")
        for cls in clf.classes_:
            cvlogistic.plot_coef_paths(clf, outpre+"_coef_path%s.png"%cls)
        print_top_motifs(final_good_motifs)
        print_top_motifs(final_good_motifs, reverse=False)
        logging.info("Writing motifs after regression")
        outmotifs = inout.ShapeMotifFile()
        outmotifs.add_motifs(final_good_motifs)
        outmotifs.write_file(outpre+"_called_motifs_after_regression.dsp", records)

    for motif in final_good_motifs:
        add_motif_metadata(this_records, motif) 
        logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
        logging.info("MI: {}".format(motif['mi']))
        logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
        logging.info("Category Entropy: {}".format(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat {} is {}".format(
                key,
                motif['enrichment'][key]
            ))
            logging.info("Enrichment for Cat {} is {}".format(
                key,
                two_way_to_log_odds(motif['enrichment'][key])
            ))
    logging.info("Generating initial heatmap for passing motifs")
    if len(final_good_motifs) > 25:
        logging.info("Only plotting first 25 motifs")
        enrich_hm = smv.EnrichmentHeatmap(final_good_motifs[:25])
    else:
        enrich_hm = smv.EnrichmentHeatmap(final_good_motifs)

    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_before_hm.txt")
    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_before_hm.pdf")
        enrich_hm.display_motifs(outpre+"motif_before_hm.pdf")
    if args.optimize:
        logging.info("Optimizing motifs using {} processors".format(args.p))
        final_motifs = mp_optimize_motifs(
            final_good_motifs,
            other_records,
            args.optimize_perc,
            p=args.p,
        )
        if args.optimize_perc != 1:
            logging.info("Testing final optimized motifs on full database")
            for i,this_entry in enumerate(final_motifs):
                logging.info("Computing MI for motif {}".format(i))
                this_discrete = generate_peak_vector(
                    other_records,
                    this_entry['motif'],
                    this_entry['threshold'],
                    args.rc,
                )
                this_entry['mi'] = other_records.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = other_records.shannon_entropy()
                this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
    else:
        if args.motif_perc != 1:
            logging.info("Testing final optimized motifs on held out database")
            for i,this_entry in enumerate(final_good_motifs):
                logging.info("Computing MI for motif {}".format(i))
                this_discrete = generate_peak_vector(
                    other_records,
                    this_entry['motif'],
                    this_entry['threshold'],
                    args.rc,
                )
                this_entry['mi'] = other_records.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = other_records.shannon_entropy()
                this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
        final_motifs = final_good_motifs

    logging.info(
        "Filtering motifs by Conditional MI using {} as a cutoff".format(args.mi_perc)
    )
    novel_motifs = filter_motifs(
        final_motifs,
        other_records,
        args.mi_perc,
    )

    if args.debug:
        print_top_motifs(novel_motifs)
        print_top_motifs(novel_motifs, reverse=False)
    logging.info("{} motifs survived".format(len(novel_motifs)))
    for i, motif in enumerate(novel_motifs):
        logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
        logging.info("MI: {}".format(motif['mi']))
        if args.infoz > 0:
            logging.info("Calculating Z-score for motif {}".format(i))
            # calculate zscore
            zscore, passed = info_zscore(
                motif['discrete'],
                other_records.get_values(),
                args.infoz,
            )
            motif['zscore'] = zscore
            logging.info("Z-score: {}".format(motif['zscore']))
        if args.infoz > 0 and args.inforobust > 0:
            logging.info("Calculating Robustness for motif {}".format(i))
            num_passed = info_robustness(
                motif['discrete'],
                other_records.get_values(), 
                args.infoz,
                args.inforobust,
                args.fracjack,
            )
            motif['robustness'] = "{}/{}".format(num_passed,args.inforobust)
            logging.info("Robustness: {}".format(motif['robustness']))
        logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
        logging.info("Category Entropy: {}".format(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat {} is {}".format(
                key,
                motif['enrichment'][key]
            ))
            logging.info("Enrichment for Cat {} is {}".format(
                key,
                two_way_to_log_odds(motif['enrichment'][key])
            ))
        if args.optimize:
            logging.info("Optimize Success?: {}".format(motif['opt_success']))
            logging.info("Optimize Message: {}".format(motif['opt_message']))
            logging.info("Optimize Iterations: {}".format(motif['opt_iter']))
    logging.info("Generating final heatmap for motifs")
    enrich_hm = smv.EnrichmentHeatmap(novel_motifs)
    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_after_hm.txt")

    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_after_hm.pdf")
        enrich_hm.display_motifs(outpre+"_motif_after_hm.pdf")
        if args.optimize:
            logging.info("Plotting optimization for final motifs")
            enrich_hm.plot_optimization(outpre+"_optimization.pdf")

    logging.info("Writing final motifs")
    outmotifs = inout.ShapeMotifFile()
    outmotifs.add_motifs(novel_motifs)
    outmotifs.write_file(outpre+"_called_motifs.dsp", records)
    #final = opt.minimize(lambda x: -optimize_mi(x, data=records, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=records), motif_to_optimize)
    #logging.info(final)
