import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
import shapemotifvis as smv
import multiprocessing as mp

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
    this_data = data
    #this_data = data.random_subset_by_class(sample_perc)
    this_discrete = generate_peak_vector(this_data, param_vec[:-1], threshold)
    this_mi = this_data.mutual_information(this_discrete)
    if info["NFeval"]%10 == 0:
        info["value"].append(this_mi)
        info["eval"].append(info["NFeval"])
    info["NFeval"] += 1
    return -this_mi

def generate_peak_vector(data, motif_vec, threshold):
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
    for this_seq in data.iterate_through_precompute():
        seq_pass = 0
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            if distance < threshold:
                seq_pass = 1
                break
        this_discrete.append(seq_pass)
    return np.array(this_discrete)

def find_initial_threshold(cats):
    """ Function to determine a reasonable starting threshold given a sample
    of the data

    Args:
        cats (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
    Returns:
        threshold (float) - a threshold that is the
                            mean(distance)-2*stdev(distance)
    """
    seeds = []
    for this_seq in cats.iterate_through_precompute():
        seeds.extend(this_seq)
    distances = []
    for seed1 in seeds:
        for seed2 in seeds:
            distances.append(seed1.distance(seed2.as_vector(cache=True), vec=True, cache=True))
    distances = np.array(distances)
    distances = distances[distances > 0]
    distances = np.log(distances)

    mean = np.mean(distances)
    stdev = np.std(distances)
    return np.exp(mean-2*stdev)

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
        logging.warning("Seeds in Cat %i: %i"%(value, np.sum(values == value)))
    return seeds

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
                                    threshold = threshold)
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
    mi_threshold = cats.shannon_entropy()*mi_threshold
    these_seeds = sorted(seeds, key=lambda x: x['mi'], reverse=True)
    top_seeds = [these_seeds[0]]
    for cand_seed in these_seeds[1:]:
        seed_pass = True
        if cand_seed['mi'] < mi_threshold:
            continue
        for good_seed in top_seeds:
            this_mi = inout.conditional_mutual_information(cats.get_values(), 
                                                 cand_seed['discrete'], 
                                                 good_seed['discrete'])
            if this_mi < mi_threshold:
                seed_pass = False
                break
        if seed_pass:
            top_seeds.append(cand_seed)
    return top_seeds

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
                         help='number of seeds to start', default=100)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--threshold_perc', type=float, default=0.05)
    parser.add_argument('--optimize_perc', type=float, default=0.1)
    parser.add_argument('--seed_perc', type=float, default=1)
    parser.add_argument('--continuous', type=int, default=None)
    parser.add_argument('--optimize', action="store_true")
    parser.add_argument('--mi_perc', type=float, default=0.01)
    parser.add_argument('--distance_metric', type=str, default="manhattan")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-o', type=str, default="motif_out_")
    parser.add_argument('-p', type=int, default=1, help="number of processors")

 
    args = parser.parse_args()
    outpre = args.o
    np.random.seed(args.seed)
    
    logging.warning("Reading in files")
    all_params = [read_parameter_file(x) for x in args.params]
    dist_met = {"manhattan": dsp.manhattan_distance, 
                "hamming": dsp.hamming_distance,
                "euclidean": dsp.euclidean_distance}
    this_dist = dist_met[args.distance_metric]
    cats = inout.SeqDatabase(names=[])
    if args.continuous is not None:
        cats.read(args.infile, float)
        logging.warning("Discretizing data")
        cats.discretize_quant(args.continuous)
    else:
        cats.read(args.infile, int)
    logging.warning("Distribution of sequences per class:")
    logging.warning(seqs_per_bin(cats))

    for name, param in zip(cats.names, cats):
        for this_param, this_param_name in zip(all_params, args.param_names):
            param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))
            param.metric = this_dist

    logging.warning("Normalizing parameters")
    if args.nonormalize:
        cats.normalize_params(inout.identity_csp)
    else:
        cats.normalize_params()
    for name in cats.center_spread.keys():
        logging.warning("%s: %s"%(name, cats.center_spread[name]))

    logging.warning("Precomputing all windows")
    cats.pre_compute_windows(args.kmer, wstart=args.ignorestart, wend=args.ignoreend)

    logging.warning("Determining initial threshold")
    if args.distance_metric == "hamming":
        threshold = 4
        logging.warning("Using %f as an initial threshold"%(threshold))
    else:
        threshold = find_initial_threshold(cats.random_subset_by_class(args.threshold_perc))
        logging.warning("Using %f as an initial threshold"%(threshold))

    all_seeds = []
    
    #greedy_threshold = threshold + 1/np.sqrt(args.kmer*len(args.params))*threshold
    greedy_threshold = threshold
    logging.warning("Greedy search for possible motifs with threshold %s"%(greedy_threshold))
    possible_motifs = greedy_search(cats, greedy_threshold, args.num_seeds)
    logging.warning("%s possible motifs"%(len(possible_motifs)))
    logging.warning("Finding MI for seeds")
    this_entry = {}

    if args.seed_perc != 1:
        this_cats = cats.random_subset_by_class(args.seed_perc)
    else:
        this_cats = cats
    logging.warning("Distribution of sequences per class for seed screening")
    logging.warning(seqs_per_bin(this_cats))
    for i,motif1 in enumerate(possible_motifs):
        if not (i % 10):
            logging.warning("Computing MI for motif %s"%i)
        this_discrete = generate_peak_vector(this_cats, motif1.as_vector(cache=True), threshold)
        this_entry['mi'] = this_cats.mutual_information(this_discrete)
        this_entry['motif_entropy'] = inout.entropy(this_discrete)
        this_entry['category_entropy'] = this_cats.shannon_entropy()
        this_entry['enrichment'] = this_cats.calculate_enrichment(this_discrete)
        this_entry['seed'] = motif1
        this_entry['threshold'] = threshold
        this_entry['discrete'] = this_discrete
        all_seeds.append(this_entry)
        this_entry = {}
    logging.warning("Filtering seeds by Conditional MI using %f as a cutoff"%(args.mi_perc*this_cats.shannon_entropy()))
    good_seeds = filter_seeds(all_seeds, this_cats, args.mi_perc)
    logging.warning("%s seeds survived"%(len(good_seeds)))
    for motif in good_seeds:
        logging.warning("Seed: %s"%(motif['seed'].as_vector(cache=True)))
        logging.warning("MI: %f"%(motif['mi']))
        logging.warning("Motif Entropy: %f"%(motif['motif_entropy']))
        logging.warning("Category Entropy: %f"%(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.warning("Two way table for cat %s is %s"%(key, motif['enrichment'][key]))
            logging.warning("Enrichment for Cat %s is %s"%(key, two_way_to_log_odds(motif['enrichment'][key])))
    logging.warning("Generating initial heatmap for passing seeds")
    if len(good_seeds) > 25:
        logging.warning("Only plotting first 25 seeds")
        enrich_hm = smv.EnrichmentHeatmap(good_seeds[:35])
    else:
        enrich_hm = smv.EnrichmentHeatmap(good_seeds)
    enrich_hm.display_enrichment(outpre+"enrichment_before_hm.pdf")
    enrich_hm.display_motifs(outpre+"motif_before_hm.pdf")
    if args.optimize:
        logging.warning("Optimizing seeds using %i processors"%(args.p))
        final_seeds = mp_optimize_seeds(good_seeds, cats, args.optimize_perc, p=args.p)
        if args.optimize_perc != 1:
            logging.warning("Testing final optimized seeds on full database")
            for i,this_entry in enumerate(final_seeds):
                logging.warning("Computing MI for motif %s"%i)
                this_discrete = generate_peak_vector(cats, this_entry['seed'].as_vector(cache=True), this_entry['threshold'])
                this_entry['mi'] = cats.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = cats.shannon_entropy()
                this_entry['enrichment'] = cats.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
        logging.warning("Filtering final seeds by BIC")
        final_good_seeds = bic_seeds(final_seeds, cats)
        if len(final_good_seeds) < 1: 
            logging.warning("No motifs found")
        logging.warning("%s seeds survived"%(len(final_good_seeds)))
        for motif in final_good_seeds:
            logging.warning("Seed: %s"%(motif['seed'].as_vector(cache=True)))
            logging.warning("MI: %f"%(motif['mi']))
            logging.warning("Motif Entropy: %f"%(motif['motif_entropy']))
            logging.warning("Category Entropy: %f"%(motif['category_entropy']))
            for key in sorted(motif['enrichment'].keys()):
                logging.warning("Two way table for cat %s is %s"%(key, motif['enrichment'][key]))
                logging.warning("Enrichment for Cat %s is %s"%(key, two_way_to_log_odds(motif['enrichment'][key])))
            logging.warning("Success?: %s"%(motif['opt_success']))
            logging.warning("Message: %s"%(motif['opt_message']))
            logging.warning("Iterations: %s"%(motif['opt_iter']))
        logging.warning("Generating final heatmap for optimized seeds")
        enrich_hm = smv.EnrichmentHeatmap(final_good_seeds)
        enrich_hm.display_enrichment(outpre+"enrichment_after_hm.pdf")
        enrich_hm.display_motifs(outpre+"motif_after_hm.pdf")
        logging.warning("Plotting optimization for final motifs")
        enrich_hm.plot_optimization(outpre+"optimization.pdf")
        logging.warning("Writing final motifs")
        outmotifs = inout.ShapeMotifFile()
        outmotifs.add_motifs(final_good_seeds)
        outmotifs.write_file(outpre+"called_motifs.dsp", cats)
    else:
        if args.optimize_perc != 1:
            logging.warning("Testing final optimized seeds on full database")
            for i,this_entry in enumerate(good_seeds):
                logging.warning("Computing MI for motif %s"%i)
                this_discrete = generate_peak_vector(cats, this_entry['seed'].as_vector(cache=True), this_entry['threshold'])
                this_entry['mi'] = cats.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = cats.shannon_entropy()
                this_entry['enrichment'] = cats.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
        logging.warning("Filtering final seeds by BIC")
        final_good_seeds = bic_seeds(good_seeds, cats)
        if len(final_good_seeds) < 1: 
            logging.warning("No motifs found")
        logging.warning("%s seeds survived"%(len(final_good_seeds)))
        logging.warning("%s seeds survived"%(len(final_good_seeds)))
        logging.warning("Writing final motifs")
        enrich_hm = smv.EnrichmentHeatmap(final_good_seeds)
        enrich_hm.display_enrichment(outpre+"enrichment_after_hm.pdf")
        enrich_hm.display_motifs(outpre+"motif_after_hm.pdf")
        outmotifs = inout.ShapeMotifFile()
        outmotifs.add_motifs(final_good_seeds)
        outmotifs.write_file(outpre+"called_motifs.dsp", cats)

    #final = opt.minimize(lambda x: -optimize_mi(x, data=cats, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=cats), motif_to_optimize)
    #logging.warning(final)
