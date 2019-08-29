import inout
import sys
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
import shapemotifvis as smv
import multiprocessing as mp
import itertools
import welfords
import copy

def make_initial_seeds(cats, wsize,wstart,wend):
    """ Function to make all possible seeds, superceded by the precompute
    all windows method in seq_database
    """
    seeds = []
    for param in cats:
        for window in param.sliding_windows(wsize, start=wstart, end=wend):
            seeds.append(window)
    return seeds

def optimize_mi(param_vec, data, sample_perc, info):
    """ Function to optimize a particular motif

    Args:
        param_vec (list) - values to optimize. Last value is threshold
        data (SeqDatabase) - full seq database
        sample_perc(float) - percentage of database to calculate MI over
        info (dict) - store number of function evals and value associated with it
                      keys must include NFeval: int, value: list, eval: list
    Returns:
        MI for the parameters over a random sample of the seq database
        as determined by 
    """
    threshold = param_vec[-1]
    this_data = data                                                            #Global, fix later to not be global
    #this_data = data.random_subset_by_class(sample_perc)                       v
    this_discrete = generate_peak_vector(this_data, param_vec[:-1], threshold, args.rc)
    this_mi = this_data.mutual_information(this_discrete)
    if info["NFeval"]%10 == 0:
        info["value"].append(this_mi)
        info["eval"].append(info["NFeval"])
    info["NFeval"] += 1
    return -this_mi

def generate_peak_vector(data, motif, threshold, rc=False):
    """ Function to calculate the sequences that have a match to a motif
    Args:
        data (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
        motif_vec(np.array) - numpy array containing motif vector
        threshold(float) - a distance threshold to be considered a match
    Returns:
        discrete (np.array) - a numpy array of 1 or 0 for each sequence where
                              1 is a match and 0 is not.
    """
    this_discrete = []
    motif_vec = motif.as_vector(cache=True)
    if rc:
        motif.rev_comp()
        motif_vec_rc = motif.as_vector()
        motif.rev_comp()

    for this_seq in data.iterate_through_precompute():
        seq_pass = False
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            if distance < threshold:
                seq_pass = True
                break
        if rc:
            for this_motif in this_seq:
                distance = this_motif.distance(motif_vec_rc, vec=True, cache=True)
                if distance < threshold:
                    seq_pass = True
                    break

        this_discrete.append(seq_pass)
    return np.array(this_discrete)

def find_initial_threshold(cats, seeds_per_seq=1, max_seeds = 10000):
    """ Function to determine a reasonable starting threshold given a sample
    of the data

    Args:
        cats (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
    Returns:
        threshold (float) - a threshold that is the
                            mean(distance)-2*stdev(distance))
    """
    # calculate stdev and mean using welford's algorithm
    online_mean = welfords.Welford()
    cats_shuffled = cats.shuffle()
    total_seeds = []
    seed_counter = 0
    # get a set of seeds to run against each other
    for i, seq in enumerate(cats_shuffled.iterate_through_precompute()):

        # sample in a random order from the sequence
        rand_order = np.random.permutation(len(seq))
        curr_seeds_per_seq = 0
        for index in rand_order:
            motif = seq[index]
            if curr_seeds_per_seq >= seeds_per_seq:
                break
            total_seeds.append(motif)
            curr_seeds_per_seq += 1
            seed_counter += 1
        if seed_counter >= max_seeds:
            break

    logging.info("Using %s random seeds to determine threshold from pairwise distances"%(len(total_seeds)))
    for i, seedi in enumerate(total_seeds):
        for j, seedj in enumerate(total_seeds):
            newval = seedi.distance(seedj.as_vector(cache=True), vec=True, cache=True)
            if i >= j:
                continue
            else:
                newval = seedi.distance(seedj.as_vector(cache=True), vec=True, cache=True)
                online_mean.update(newval)

    mean = online_mean.final_mean()
    stdev = online_mean.final_stdev()
    
    logging.info("Threshold mean: %s and stdev %s"%(mean, stdev))
    return mean, stdev

def seqs_per_bin(cats):
    """ Function to determine how many sequences are in each category

    Args:
        cats (SeqDatabase) - database to calculate over
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    string = ""
    for value in np.unique(cats.values):
        string += "\nCat %i: %i"%(value, np.sum(np.array(cats.values) ==  value))
    return string

def two_way_to_log_odds(two_way):
    """ Function to determine the log odds from a two way table

    Args:
        two_way (list)- a list of 0-cat1Truecat2True 1-cat1Falsecat2True 
                                  2-cat1Truecat2False 3-cat1Falsecat2False
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    num = np.array(two_way[0], dtype=float)/np.array(two_way[1],dtype=float)
    denom = np.array(two_way[2], dtype=float)/np.array(two_way[3],dtype=float)
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

class MotifMatch(Exception):
    """ Exception class used for greedy search. To be raised when a motif
    match is found
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return "Distance is %f"%self.value

def greedy_search(cats, threshold = 10, number=1000):
    """ Function to find initial seeds by a greedy search

    Prints the number of seeds per class to the logger

    Args:
        cats (inout.SeqDatabase) - input data, must have motifs already 
                                   precomputed
        threshold (float) - threshold for considering a motif a match
        number (int) - number of seeds to stop near 
    Returns:
        seeds (list) - a list of dsp.ShapeParamSeq objects to be considered
    """
    seeds = []
    values = []
    cats_shuffled = cats.shuffle()
    for i,seq in enumerate(cats_shuffled.iterate_through_precompute()):
        if(len(seeds) >= number):
            break
        for motif in seq:
            try:
                for motif2 in seeds:
                    distance = motif2.distance(motif.as_vector(), vec=True, cache=True)
                    if distance < threshold:
                        raise MotifMatch(distance)
                seeds.append(motif)
                values.append(cats_shuffled.values[i])
            except MotifMatch as e:
                continue
    values = np.array(values)
    for value in np.unique(values):
        logging.info("Seeds in Cat %i: %i"%(value, np.sum(values == value)))
    return seeds

def greedy_search2(cats, threshold = 10, number=1000, seeds_per_seq = 1, rc=False, prev_seeds=None):
    """ Function to find initial seeds by a greedy search

    Prints the number of seeds per class to the logger

    Args:
        cats (inout.SeqDatabase) - input data, must have motifs already 
                                   precomputed
        threshold (float) - threshold for considering a motif a match
        number (int) - number of seeds to stop near 
    Returns:
        seeds (list) - a list of dsp.ShapeParamSeq objects to be considered
    """
    if prev_seeds:
        seeds = prev_seeds
    else:
        seeds = []
    values = []
    cats_shuffled = cats.shuffle()
    for i,seq in enumerate(cats_shuffled.iterate_through_precompute()):
        if i % 500 == 0:
            logging.info("Greedy search on seq %s"%(i))
        if(len(seeds) >= number):
            break
        # sample in a random order from the sequence
        rand_order = np.random.permutation(len(seq))
        curr_seeds_per_seq=0
        for index in rand_order:
            motif = seq[index]
            if curr_seeds_per_seq >= seeds_per_seq:
                break
            try:
                for j,motif2 in enumerate(seeds):
                    distance = motif2.distance(motif.as_vector(), vec=True, cache=True)
                    if distance < threshold:
                        coin_flip = np.random.randint(0,2)
                        if coin_flip:
                            seeds[j] = motif
                        raise MotifMatch(distance)
                seeds.append(motif)
                curr_seeds_per_seq += 1
                values.append(cats_shuffled.values[i])
            except MotifMatch as e:
                continue
    values = np.array(values)
    for value in np.unique(values):
        logging.info("Seeds in Cat %i: %i"%(value, np.sum(values == value)))
    return seeds

def evaluate_seeds(cats, motifs, threshold_match, rc):
    """ Function to evaluate a set of seeds and return the results in a list

    Args:
        cats - full sequence database. This is read only
        motifs - set of motifs, again read only
        threshold_match - distance threshold to be considered for a match
        rc - test the reverse complement of the motif or not
    """
    seeds = []
    for motif in motifs:
        this_entry = {}
        this_discrete = generate_peak_vector(cats, motif, threshold_match, rc=rc)
        this_entry['mi'] = cats.mutual_information(this_discrete)
        this_entry['seed'] = motif
        this_entry['discrete'] = this_discrete
        this_entry['threshold'] = threshold_match
        seeds.append(this_entry)
    return seeds

def add_seed_metadata(cats, seed):
    seed['motif_entropy'] = inout.entropy(seed['discrete'])
    seed['category_entropy'] = cats.shannon_entropy()
    seed['enrichment'] = cats.calculate_enrichment(seed['discrete'])

def print_top_seeds(seeds, n= 5, reverse=True):
    sorted_seeds = sorted(seeds, key=lambda x: x['mi'], reverse=reverse)
    if reverse:
        logging.debug("Printing top %s seeds."%(n))
    else:
        logging.debug("Printing bottom %s seeds."%(n))

    for seed in sorted_seeds[0:n]:
        logging.debug("Seed MI: %s\n Seed Mem: %s\n%s"%(seed['mi'],seed['seed'], seed['seed'].as_vector()))

def save_mis(seeds, out_pre):
    all_mis = []
    for seed in seeds:
        all_mis.append(seed['mi'])
    np.savetxt(out_pre+".txt", all_mis)


def mp_optimize_seeds_helper(args):
    """ Helper function to allow seed optimization to be multiprocessed
    
    Args:
        args (list) - list containing starting seed dict, category, and sample
                      percentage
    
    Returns:
        final_seed_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """
    seed, data, sample_perc = args
    #this_data = data.random_subset_by_class(sample_perc)
    final_seed_dict = {}
    func_info = {"NFeval":0, "eval":[], "value":[]}
    motif_to_optimize = list(seed['seed'].as_vector(cache=True))
    motif_to_optimize.append(seed['threshold'])
    final_opt = opt.minimize(optimize_mi,motif_to_optimize, 
                             args=(data, sample_perc, func_info), 
                             method="nelder-mead")
    final = final_opt['x']
    threshold = final[-1]
    final_seed = dsp.ShapeParams()
    final_seed.from_vector(seed['seed'].names, final[:-1])
    final_seed_dict['seed'] = final_seed
    discrete = generate_peak_vector(data, final_seed.as_vector(cache=True), 
                                    threshold = threshold, rc=args.rc)
    final_seed_dict['threshold'] = threshold
    final_seed_dict['enrichment'] = data.calculate_enrichment(discrete)
    final_seed_dict['motif_entropy'] = inout.entropy(discrete)
    final_seed_dict['category_entropy'] = data.shannon_entropy()
    final_seed_dict['mi'] = data.mutual_information(discrete)
    final_seed_dict['discrete'] = discrete
    final_seed_dict['opt_success'] = final_opt['success']
    final_seed_dict['opt_message'] = final_opt['message']
    final_seed_dict['opt_iter'] = final_opt['nit']
    final_seed_dict['opt_func'] = final_opt['nfev']
    final_seed_dict['opt_info'] = func_info

    return final_seed_dict

def mp_evaluate_seeds(data, seeds, threshold_match, rc, p=1):
    """Perform seed evaluation in a multiprocessed manner

    Args:
        seeds (list) - seeds to evaluate
        data (cat) - A sequence category object
        p (int) - number of processors

    Returns:
        list of evaluated seed dictionaries
    """
    # determine how to chunk the seeds:
    seeds_chunked = [seeds[i::p] for i in range(0,p)]
    pool = mp.Pool(processes=p)
    out_seeds = pool.map(mp_evaluate_seeds_helper, 
            ((data, seeds, threshold_match, rc) for seeds in seeds_chunked))
    pool.close()
    pool.join()
    # flatten list
    final_seeds = []
    for processor in out_seeds:
        for seed in processor:
            final_seeds.append(seed)
    return final_seeds

def mp_evaluate_seeds_helper(these_args):
    """ Helper function for doing evaluation in a multiprocessed way
    
    Args:
        these_args (list) - list of args i.e. database, list of seeds,
                            match cutoff, reverse complement

    Returns:
        list of evaluated seed dictionaries
    """
    data, seeds, threshold_match, rc = these_args
    these_seeds = evaluate_seeds(data, seeds, threshold_match, rc)
    return these_seeds

        
def mp_optimize_seeds(seeds, data, sample_perc, p=1):
    """Perform seed optimization in a multiprocessed manner
    
    Args:
        seeds (list) - List containing seed dictionaries
        data (SeqDatabase) - Full dataset to train on
        sample_perc (float) - percentage of data to sample for optimization
    
    Returns:
        final_seed_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """
    pool = mp.Pool(processes=p)
    this_data = data.random_subset_by_class(sample_perc)
    final_seeds = pool.map(mp_optimize_seeds_helper, 
                           ((seed, this_data, sample_perc) for seed in seeds))
    pool.close()
    pool.join()
    return final_seeds

def filter_seeds(seeds, cats, mi_threshold):
    """ Select initial seeds through conditional mutual information

    Args:
        seeds (list of dicts) - list of motif dictionaries
        cats (SeqDatabase Class) - sequences motifs are compared against
        mi_threshold (float) - percentage of total entropy CMI must be >
    
    Returns:
        final_seeds (list of dicts) - list of passing motif dictionaries
    """
    these_seeds = sorted(seeds, key=lambda x: x['mi'], reverse=True)
    top_seeds = [these_seeds[0]]
    for cand_seed in these_seeds[1:]:
        seed_pass = True
        for good_seed in top_seeds:
            cmi = inout.conditional_mutual_information(cats.get_values(), 
                                                 cand_seed['discrete'], 
                                                 good_seed['discrete'])

            mi_btwn_seeds = inout.mutual_information(cand_seed['discrete'], 
                    good_seed['discrete'])
            if mi_btwn_seeds == 0:
                ratio = 0
            else:
                ratio = cmi/mi_btwn_seeds
            if ratio < mi_threshold:
                seed_pass = False
                break
        if seed_pass:
            top_seeds.append(cand_seed)
    return top_seeds

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
    for i in xrange(n):
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
    for i in xrange(r):
        jk_selector = np.random.permutation(len(vec1))[0:num_to_use]
        zscore, passed = info_zscore(vec1[jk_selector], vec2[jk_selector], n=n)
        if passed:
            num_passed += 1
    return num_passed

def aic_seeds(seeds, cats):
    """ Select final seeds through AIC

    Args:
        seeds (list of dicts) - list of final motif dictionaries
        cats (SeqDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_seeds (list of dicts) - list of passing motif dictionaries
    """

    # get number of parameters based on length of motif vector
    delta_k = len(seeds[0]['seed'].as_vector(cache=True))
    # get number of sequences
    n = len(cats)
    # sort seeds by mutual information
    these_seeds = sorted(seeds, key=lambda x: x['mi'], reverse=True)
    # Make sure first seed passes AIC
    if (2*delta_k - 2*n*these_seeds[0]['mi']) < 0:
        top_seeds = [these_seeds[0]]
    else:
        return []
    # loop through candidate seeds
    for cand_seed in these_seeds[1:]:
        seed_pass = True
        # if the total MI for this seed doesn't pass AIC skip it
        if (2*delta_k - 2*n*cand_seed['mi']) > 0:
            continue
        for good_seed in top_seeds:
            # check the conditional mutual information for this seed with
            # each of the chosen seeds
            this_mi = inout.conditional_mutual_information(cats.get_values(), 
                                             cand_seed['discrete'], 
                                             good_seed['discrete'])
            # if candidate seed doesn't improve model as added to each of the 
            # chosen seeds, skip it
            if (2*delta_k- 2*n*this_mi) > 0:
                seed_pass = False
                break
        if seed_pass:
            top_seeds.append(cand_seed)
    return top_seeds

def bic_seeds(seeds, cats):
    """ Select final seeds through BIC

    Args:
        seeds (list of dicts) - list of final motif dictionaries
        cats (SeqDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_seeds (list of dicts) - list of passing motif dictionaries
    """
    delta_k = len(seeds[0]['seed'].as_vector(cache=True))
    n = len(cats)
    these_seeds = sorted(seeds, key=lambda x: x['mi'], reverse=True)
    if 2*delta_k*np.log2(n) - 2*n*these_seeds[0]['mi'] < 0:
        top_seeds = [these_seeds[0]]
    else:
        return []
    for cand_seed in these_seeds[1:]:
        seed_pass = True
        if 2*delta_k*np.log2(n) - 2*n*cand_seed['mi'] > 0:
            continue
        for good_seed in top_seeds:
            this_mi = inout.conditional_mutual_information(cats.get_values(), 
                                             cand_seed['discrete'], 
                                             good_seed['discrete'])
            if 2*delta_k*np.log2(n) - 2*n*this_mi > 0:
                seed_pass = False
                break
        if seed_pass:
            top_seeds.append(cand_seed)
    return top_seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', action='store', type=str,
                         help='inputfile with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--kmer', type=int,
                         help='kmer size to search for', default=15)
    parser.add_argument('--ignorestart', type=int,
                         help='# bp to ignore at start of each sequence', default=2)
    parser.add_argument('--ignoreend', type=int,
                         help='# bp to ignore at end of each sequence', default=2)
    parser.add_argument('--num_seeds', type=int,
                         help='cutoff for number of seeds to test. Default=1000', default=1000)
    parser.add_argument('--seeds_per_seq', type=int,
                         help='max number of seeds to come from a single sequence. Default=1', default=1)
    parser.add_argument('--seeds_per_seq_thresh', type=int,
                         help='max number of seeds to come from a single sequence. Default=1', default=1)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--search_method', type=str, default="greedy", help="search method for initial seeds. Options: greedy, brute")
    parser.add_argument('--threshold_perc', type=float, default=0.05,
            help="fraction of data to determine threshold on. Default=0.05")
    parser.add_argument('--threshold_seeds', type=float, default=2.0, 
            help="std deviations below mean for seed finding. Default=2.0")
    parser.add_argument('--threshold_match', type=float, default=2.0, 
            help="std deviations below mean for match. Default=2.0")
    parser.add_argument('--optimize_perc', type=float, default=0.1, 
            help="fraction of data to optimize on. Default=0.1")
    parser.add_argument('--seed_perc', type=float, default=1,
            help="fraction of data to EVALUATE seeds on. Default=1")
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--optimize', action="store_true",
            help="optimize seeds with Nelder Mead?")
    parser.add_argument('--mi_perc', type=float, default=5,
            help="ratio of CMI/MI to include an additional seed. default=5")
    parser.add_argument('--infoz', type=int, default=2000, 
            help="Calculate Z-score for final motif MI with n data permutations. default=2000. Turn off by setting to 0")
    parser.add_argument('--inforobust', type=int, default=10, 
            help="Calculate robustness of final motif with x jacknifes. Default=10. Requires infoz to be > 0.")
    parser.add_argument('--fracjack', type=int, default=0.3, 
            help="Fraction of data to hold out in jacknifes. Default=0.3.")
    parser.add_argument('--distance_metric', type=str, default="manhattan",
            help="distance metric to use, manhattan is the only supported one for now")
    parser.add_argument('--seed', type=int, default=None,
            help="set the random seed, default=None, based on system time")
    parser.add_argument('--rc', action="store_true",
            help="search the reverse complement with each seed as well?")
    parser.add_argument('-o', type=str, default="motif_out_")
    parser.add_argument('-p', type=int, default=1, help="number of processors")
    parser.add_argument("--debug", action="store_true",
            help="print debugging information to stderr")
    parser.add_argument('--txt_only', action='store_true', help="output only txt files?")
    
    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level) 
    outpre = args.o
    # choose a random seed
    if args.seed:
        np.random.seed(args.seed)
    
    logging.info("Reading in files")
    # read in the fasq files containing parameter information
    all_params = [read_parameter_file(x) for x in args.params]
    # possible distance metrics that could be used
    dist_met = {"manhattan": dsp.manhattan_distance, 
                "hamming": dsp.hamming_distance,
                "euclidean": dsp.euclidean_distance}
    # store the distance metric chosen
    this_dist = dist_met[args.distance_metric]
    # create an empty sequence database to store the sequences in
    cats = inout.SeqDatabase(names=[])
    
    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        cats.read(args.infile, float)
        logging.info("Discretizing data")
        cats.discretize_quant(args.continuous)
    else:
        cats.read(args.infile, int)
    logging.info("Distribution of sequences per class:")
    logging.info(seqs_per_bin(cats))

    # add parameter values for each sequence
    for name, param in zip(cats.names, cats):
        for this_param, this_param_name in zip(all_params, args.param_names):
            param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))
            param.metric = this_dist

    logging.info("Normalizing parameters")
    if args.nonormalize:
        cats.determine_center_spread(inout.identity_csp)
    else:
        cats.determine_center_spread()
        cats.normalize_params()
    for name in cats.center_spread.keys():
        logging.info("%s: %s"%(name, cats.center_spread[name]))

    logging.info("Precomputing all windows")
    cats.pre_compute_windows(args.kmer, wstart=args.ignorestart, wend=args.ignoreend)

    logging.info("Determining initial threshold")
    if args.distance_metric == "hamming":
        threshold_match = 4
        logging.info("Using %f as an initial match threshold"%(threshold_match))
    else:
        mean,stdev = find_initial_threshold(cats.random_subset_by_class(args.threshold_perc), args.seeds_per_seq_thresh)
        threshold_seeds = max(mean - args.threshold_seeds*stdev, 0)
        threshold_match = max(mean - args.threshold_match*stdev, 0)
        logging.info("Using %f as an initial match threshold"%(threshold_match))
    if args.search_method == "greedy":
        logging.info("Greedy search for possible motifs with threshold %s"%(threshold_seeds))
        possible_motifs = greedy_search2(cats, threshold_seeds, args.num_seeds, args.seeds_per_seq)
    else:
        logging.info("Testing all seeds by brute force")
        # double for loop list comprehension
        possible_motifs = [motif for a_seq in cats.iterate_through_precompute() for motif in a_seq]

    logging.info("%s possible seeds"%(len(possible_motifs)))
    logging.info("Finding MI for seeds")

    if args.seed_perc != 1:
        this_cats = cats.random_subset_by_class(args.seed_perc)
    else:
        this_cats = cats
    logging.info("Distribution of sequences per class for seed screening")
    logging.info(seqs_per_bin(this_cats))
    logging.info("Evaluating %s seeds over %s processor(s)"%(len(possible_motifs), args.p))

    all_seeds = mp_evaluate_seeds(this_cats, possible_motifs, threshold_match, args.rc, p=args.p)
    #all_seeds = evaluate_seeds(this_cats, possible_motifs, threshold_match, args.rc)
    if args.debug:
        print_top_seeds(all_seeds)
        print_top_seeds(all_seeds, reverse=False)
        save_mis(all_seeds, args.o+"_all_seeds_mi")
    logging.info("Filtering seeds by Conditional MI using %f as a cutoff"%(args.mi_perc))
    novel_seeds = filter_seeds(all_seeds, this_cats, args.mi_perc)
    logging.info("%s seeds survived"%(len(novel_seeds)))
    if args.debug:
        print_top_seeds(novel_seeds)
        print_top_seeds(novel_seeds, reverse=False)
    logging.info("Filtering seeds by AIC individually")
    good_seeds = []
    for seed in novel_seeds:
        passed = aic_seeds([seed], this_cats)
        if len(passed) > 0:
            good_seeds.append(seed) 
    if len(good_seeds) < 1: 
        logging.info("No motifs found")
        sys.exit()
    if args.debug:
        print_top_seeds(good_seeds)
        print_top_seeds(good_seeds, reverse=False)
    logging.info("%s seeds survived"%(len(good_seeds)))
    for motif in good_seeds:
        add_seed_metadata(this_cats, motif) 
        logging.info("Seed: %s"%(motif['seed'].as_vector(cache=True)))
        logging.info("MI: %f"%(motif['mi']))
        logging.info("Motif Entropy: %f"%(motif['motif_entropy']))
        logging.info("Category Entropy: %f"%(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat %s is %s"%(key, motif['enrichment'][key]))
            logging.info("Enrichment for Cat %s is %s"%(key, two_way_to_log_odds(motif['enrichment'][key])))
    logging.info("Generating initial heatmap for passing seeds")
    if len(good_seeds) > 25:
        logging.info("Only plotting first 25 seeds")
        enrich_hm = smv.EnrichmentHeatmap(good_seeds[:25])
    else:
        enrich_hm = smv.EnrichmentHeatmap(good_seeds)

    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_before_hm.txt")
    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_before_hm.pdf")
        enrich_hm.display_motifs(outpre+"motif_before_hm.pdf")
    if args.optimize:
        logging.info("Optimizing seeds using %i processors"%(args.p))
        final_seeds = mp_optimize_seeds(good_seeds, cats, args.optimize_perc, p=args.p)
        if args.optimize_perc != 1:
            logging.info("Testing final optimized seeds on full database")
            for i,this_entry in enumerate(final_seeds):
                logging.info("Computing MI for motif %s"%i)
                this_discrete = generate_peak_vector(cats, this_entry['seed'], this_entry['threshold'], args.rc)
                this_entry['mi'] = cats.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = cats.shannon_entropy()
                this_entry['enrichment'] = cats.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
    else:
        if args.seed_perc != 1:
            logging.info("Testing final optimized seeds on full database")
            for i,this_entry in enumerate(good_seeds):
                logging.info("Computing MI for motif %s"%i)
                this_discrete = generate_peak_vector(cats, this_entry['seed'], this_entry['threshold'], args.rc)
                this_entry['mi'] = cats.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = cats.shannon_entropy()
                this_entry['enrichment'] = cats.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
        final_seeds = good_seeds
    logging.info("Filtering final seeds by BIC")
    final_good_seeds = bic_seeds(final_seeds, cats)
    if len(final_good_seeds) < 1: 
        logging.info("No motifs found")
        sys.exit()
    logging.info("%s seeds survived"%(len(final_good_seeds)))
    if args.debug:
        print_top_seeds(final_good_seeds)
        print_top_seeds(final_good_seeds, reverse=False)
    for i, motif in enumerate(final_good_seeds):
        logging.info("Seed: %s"%(motif['seed'].as_vector(cache=True)))
        logging.info("MI: %f"%(motif['mi']))
        if args.infoz > 0:
            logging.info("Calculating Z-score for motif %s"%i)
            # calculate zscore
            zscore, passed = info_zscore(motif['discrete'], cats.get_values(), args.infoz)
            motif['zscore'] = zscore
            logging.info("Z-score: %f"%(motif['zscore']))
        if args.infoz > 0 and args.inforobust > 0:
            logging.info("Calculating Robustness for motif %s"%i)
            num_passed = info_robustness(motif['discrete'], cats.get_values(), 
                    args.infoz, args.inforobust, args.fracjack)
            motif['robustness'] = "%s/%s"%(num_passed,args.inforobust)
            logging.info("Robustness: %s"%(motif['robustness']))
        logging.info("Motif Entropy: %f"%(motif['motif_entropy']))
        logging.info("Category Entropy: %f"%(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat %s is %s"%(key, motif['enrichment'][key]))
            logging.info("Enrichment for Cat %s is %s"%(key, two_way_to_log_odds(motif['enrichment'][key])))
        if args.optimize:
            logging.info("Optimize Success?: %s"%(motif['opt_success']))
            logging.info("Optimize Message: %s"%(motif['opt_message']))
            logging.info("Optimize Iterations: %s"%(motif['opt_iter']))
    logging.info("Generating final heatmap for seeds")
    enrich_hm = smv.EnrichmentHeatmap(final_good_seeds)
    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_after_hm.txt")
    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_after_hm.pdf")
        enrich_hm.display_motifs(outpre+"_motif_after_hm.pdf")
        if args.optimize:
            logging.info("Plotting optimization for final motifs")
            enrich_hm.plot_optimization(outpre+"_optimization.pdf")
    logging.info("Writing final motifs")
    outmotifs = inout.ShapeMotifFile()
    outmotifs.add_motifs(final_good_seeds)
    outmotifs.write_file(outpre+"_called_motifs.dsp", cats)
    #final = opt.minimize(lambda x: -optimize_mi(x, data=cats, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=cats), motif_to_optimize)
    #logging.info(final)
