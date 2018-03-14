import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
import shapemotifvis as smv
import multiprocessing as mp
import copy
np.random.seed(1234)

def make_initial_seeds(cats, wsize,wstart,wend):
    """ Function to make all possible seeds, superceded by the precompute
    all windows method in seq_database
    """
    seeds = []
    for param in cats:
        for window in param.sliding_windows(wsize, start=wstart, end=wend):
            seeds.append(window)
    return seeds

def optimize_mi(param_vec, data, sample_perc):
    """ Function to optimize a particular motif

    Args:
        param_vec (list) - values to optimize. Last value is threshold
        data (SeqDatabase) - full seq database
        sample_perc(float) - percentage of database to calculate MI over
    Returns:
        MI for the parameters over a random sample of the seq database
        as determined by 
    """
    threshold = param_vec[-1]
    this_data = data.random_subset(sample_perc)
    this_discrete = generate_peak_vector(this_data, param_vec[:-1], threshold)
    this_mi = this_data.mutual_information(this_discrete)
    return this_mi

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
    seed, data, sample_perc = args
    final_seed_dict = {}
    motif_to_optimize = list(seed['seed'].as_vector(cache=True))
    motif_to_optimize.append(seed['threshold'])
    final = opt.minimize(lambda x: -optimize_mi(x, data=cats, sample_perc=sample_perc),
            motif_to_optimize, method="nelder-mead")
    final = final['x']
    threshold = final[-1]
    final_seed = dsp.ShapeParams()
    final_seed.from_vector(seed['seed'].names, final[:-1])
    final_seed_dict['seed'] = final_seed
    discrete = generate_peak_vector(data, final_seed.as_vector(cache=True), 
                                    threshold = threshold)
    final_seed_dict['threshold'] = threshold
    final_seed_dict['enrichment'] = data.calculate_enrichment(discrete)
    final_seed_dict['entropy'] = data.shannon_entropy(discrete)
    final_seed_dict['mi'] = data.mutual_information(discrete)
    return final_seed_dict
        
def mp_optimize_seeds(seeds, data, sample_perc, p=1):
    pool = mp.Pool(processes=p)
    final_seeds = pool.map(mp_optimize_seeds_helper, 
                           ((seed, data, sample_perc) for seed in seeds))
    pool.close()
    pool.join()
    return final_seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', action='store', type=str,
                         help='inputfile with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--windowsize', type=int,
                         help='window_size to use', default=15)
    parser.add_argument('--windowstart', type=int,
                         help='window_start to use', default=2)
    parser.add_argument('--windowend', type=int,
                         help='window_start to use', default=None)
    parser.add_argument('--num_seeds', type=int,
                         help='number of seeds to start', default=100)
    parser.add_argument('--threshold_perc', type=float, default=0.05)
    parser.add_argument('--optimize_perc', type=float, default=0.1)
    parser.add_argument('--seed_perc', type=float, default=1)
    parser.add_argument('--continuous', type=int, default=None)
    parser.add_argument('-o', type=str, default="motif_out_")
    parser.add_argument('-p', type=int, default=1, help="number of processors")

 
    args = parser.parse_args()
    outpre = args.o
    
    logging.warning("Reading in files")
    all_params = [read_parameter_file(x) for x in args.params]
    cats = inout.SeqDatabase(names=[])
    if args.continuous is not None:
        cats.read(args.infile, float)
        logging.warning("Discretizing data")
        logging.warning(cats.values[0:10])
        cats.discretize_quant(args.continuous)
        logging.warning(cats.values[0:10])
    else:
        cats.read(args.infile, int)
    logging.warning("Distribution of sequences per class:")
    logging.warning(seqs_per_bin(cats))

    for name, param in zip(cats.names, cats):
        for this_param, this_param_name in zip(all_params, args.param_names):
            param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))

    logging.warning("Normalizing parameters")
    cats.normalize_params()

    logging.warning("Precomputing all windows")
    cats.pre_compute_windows(args.windowsize, wstart=args.windowstart, wend=args.windowend)

    logging.warning("Determining inital threshold")
    threshold = find_initial_threshold(cats.random_subset(args.threshold_perc))
    logging.warning("Using %f as an initial threshold"%(threshold))

    all_seeds = []

    #for motif1 in [possible_motifs[x] for x in np.random.randint(0,high=len(possible_motifs), size=100)]:
    logging.warning("Greedy search for possible motifs")
    possible_motifs = greedy_search(cats, 2*threshold, args.num_seeds)
    logging.warning("%s possible motifs"%(len(possible_motifs)))
    logging.warning("Finding MI for seeds")
    this_entry = {}
    for i,motif1 in enumerate(possible_motifs):
        if not (i % 10):
            logging.warning("Computing MI for motif %s"%i)
        if args.seed_perc != 1:
            this_cats = cats.random_subset(args.seed_perc)
        else:
            this_cats = cats
        this_discrete = generate_peak_vector(this_cats, motif1.as_vector(cache=True), threshold)
        this_entry['mi'] = this_cats.mutual_information(this_discrete)
        this_entry['entropy'] = this_cats.shannon_entropy(this_discrete)
        this_entry['enrichment'] = this_cats.calculate_enrichment(this_discrete)
        this_entry['seed'] = motif1
        this_entry['threshold'] = threshold
        all_seeds.append(this_entry)
        this_entry = {}
    logging.warning("Sorting seeds")
    all_seeds = sorted(all_seeds, key=lambda x: x['mi'])
    motif_to_optimize = list(all_seeds[-1]['seed'].as_vector(cache=True))
    logging.warning("Calculating Enrichment for top 10 seeds")
    for motif in all_seeds[-10:]:
        logging.warning("Seed: %s"%(motif['seed'].as_vector(cache=True)))
        logging.warning("MI: %f"%(motif['mi']))
        logging.warning("Entropy: %f"%(motif['entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.warning("Enrichment for Cat %s is %s"%(key, two_way_to_log_odds(motif['enrichment'][key])))
    logging.warning("Generating initial heatmap for top 10 seeds")
    enrich_hm = smv.EnrichmentHeatmap(all_seeds[-10:])
    enrich_hm.display_enrichment(outpre+"enrichment_before_hm.png")
    enrich_hm.display_motifs(outpre+"motif_before_hm.png")

    logging.warning("Optimizing seeds using %i processors"%(args.p))
    final_seeds = mp_optimize_seeds(all_seeds[-10:], cats, args.optimize_perc, p=args.p)
    logging.warning("Generating final heatmap for optimized seeds")
    enrich_hm = smv.EnrichmentHeatmap(final_seeds)
    enrich_hm.display_enrichment(outpre+"enrichment_after_hm.png")
    enrich_hm.display_motifs(outpre+"motif_after_hm.png")

    #final = opt.minimize(lambda x: -optimize_mi(x, data=cats, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=cats), motif_to_optimize)
    #logging.warning(final)
